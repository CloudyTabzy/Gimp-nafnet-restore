#!nafnet-gimp-python
"""GIMP 3.x plug-in for NAFNet image restoration.

Exposes two image-scoped procedures that share this single shebang alias:

- ``plug-in-nafnet-restore`` -- restore the entire active drawable.
- ``plug-in-nafnet-restore-region`` -- restore only the selected
  region (uses the selection bbox with a 64 px context padding so
  the model has surrounding pixels to inform the restoration).

Both procedures spawn a sidecar worker (Rust by default, Python
fallback) that runs NAFNet-REDS on the RGB channels and writes a
PNG; the plug-in loads the PNG back into the drawable's shadow
buffer and merges. For RGBA drawables, the original alpha is
extracted to a separate Y u8 PNG, passed to the worker via
``--alpha``, and recombined with the model's RGB output so the
alpha is preserved byte-for-byte through the sidecar round-trip.

Architecture mirrors the LaMa plug-in:

    GIMP process  (MINGW Python 3.14, gi + GEGL only)
    |
    +-- #!nafnet-gimp-python  (shebang -> user-level .interp mapping)
    |   |
    |   +-- nafnet-restore.py
    |        |
    |        +-- exports drawable to temp PNG (whole) or crops to
    |        |   selection ROI + 64 px context (region)
    |        +-- if RGBA: extracts alpha to alpha.png via
    |        |   buffer-source -> component-extract -> png-save
    |        +-- spawns worker subprocess
    |        |   (Rust default --features = ["cpu"], Python fallback)
    |        +-- loads result PNG into drawable shadow buffer
    |        +-- merge_shadow
    |
    +-- worker process  (Rust or Python, with Pillow + onnxruntime)
        ort CPU provider
        model pipeline: load PNG -> HWC f32 [0, 1] -> (Rust:
           512x512 tile + 2D tent-blend overlap; Python:
           single forward) -> NAFNet -> HWC f32 -> save PNG
        if --alpha given: read Y u8 alpha.png, combine with
           RGB output, save RGBA

NAFNet is 1:1 spatial (input H,W == output H,W) and 3-channel
RGB. RGBA inputs are split: the GIMP-side extracts the alpha
into a separate Y u8 PNG via GEGL's ``component-extract
component=alpha`` operation, the worker combines it with the
model's RGB output, and the result loads back as RGBA. The
alpha bytes are preserved byte-for-byte through the sidecar.

Diagnostics: every GEGL call and every worker subprocess is
bracketed by ``_log()`` lines in ``nafnet.log`` (in this
directory). On any failure the log pinpoints the exact step.
The file-based log is safe at both module load and runtime
(see Pitfall 16 in GIMP-plugin-common-pitfalls.md).

Worker selection: ``nafnet_config.json`` in this directory has a
``worker_kind`` field. ``"rust"`` (default when the binary is
present) uses 512x512 tiled inference + 2D tent-blend overlap
and handles any image size. ``"python"`` is the fallback
(single-pass; OOMs above ~1 Mpix). The Python worker is
selected automatically if no Rust binary is found.

Progress: while the worker is running, a daemon thread calls
``Gimp.progress_pulse()`` every 250 ms so the GIMP progress
bar shows activity. The phase text is "Running NAFNet
inference (~30 s for 2K images)..." for whole-image mode and
"Running NAFNet inference on the selection... please wait"
for region mode.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import gi

from gi.repository import Gegl, Gimp, GimpUi, GLib


PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.abspath(os.path.join(PLUGIN_DIR, "nafnet.log"))
LOG_MAX_LINES = 200

# Workers. Rust is the default when present (faster cold-start);
# Python is the fallback. Mirrors the LaMa plug-in config.
RUST_WORKER_BINARY = os.path.abspath(
    os.path.join(PLUGIN_DIR, "nafnet-worker_rust.exe")
)
WORKER_SCRIPT = os.path.abspath(os.path.join(PLUGIN_DIR, "nafnet_worker.py"))
MODEL_PATH = os.path.abspath(
    os.path.join(PLUGIN_DIR, "nafnet-REDS-width64_v1.onnx")
)
CONFIG_PATH = os.path.abspath(os.path.join(PLUGIN_DIR, "nafnet_config.json"))

# Context padding (in original-resolution pixels) added around the
# selection bbox for the region-only mode. The model uses surrounding
# pixels to inform the restoration (especially for deblurring), so
# a few dozen pixels of context is necessary to avoid a visible
# seam at the edge of the restored region. 64 px is a reasonable
# default; if the selection is at the very edge of the image, the
# worker pipeline (or the GIMP-side glue) clips to the image bounds.
SELECTION_CONTEXT_PX = 64

WORKER_TIMEOUT_SECONDS = 600  # large images with tiling can be slow
WORKER_POLL_INTERVAL_SECONDS = 0.25


def _log(msg: str) -> None:
    """Append a timestamped line to the plug-in log file.

    GIMP does not relay plug-in stdout/stderr to any visible console
    in GUI mode, so the log file is the only persistent record of
    what happened. The log is rotated to its most recent
    ``LOG_MAX_LINES`` entries on every write so a long install does
    not produce an unbounded file.
    """
    try:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                existing = f.read().splitlines()
        except OSError:
            existing = []
        existing.append(line.rstrip("\n"))
        if len(existing) > LOG_MAX_LINES:
            existing = existing[-LOG_MAX_LINES:]
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(existing) + "\n")
    except OSError:
        pass


def _normalize_python_path(path, relative_to=None):
    if not isinstance(path, str) or not path.strip():
        return None
    path = path.strip().strip('"')
    path = os.path.expanduser(os.path.expandvars(path))
    if not os.path.isabs(path) and relative_to:
        path = os.path.join(relative_to, path)
    return os.path.abspath(path)


def _configured_python():
    if not os.path.isfile(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, ValueError, TypeError):
        return None
    return _normalize_python_path(config.get("worker_python"), PLUGIN_DIR)


def find_worker_python():
    """Find the worker interpreter in deterministic preference order."""
    candidates = []
    configured = _configured_python()
    if configured:
        candidates.append(configured)

    env_python = _normalize_python_path(os.environ.get("NAFNET_WORKER_PYTHON"))
    if env_python:
        candidates.append(env_python)

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            for minor in range(14, 9, -1):
                candidates.append(
                    os.path.join(
                        local_app_data,
                        "Programs",
                        "Python",
                        f"Python3{minor}",
                        "python.exe",
                    )
                )

    for command in ("python", "python3"):
        discovered = shutil.which(command)
        if discovered:
            candidates.append(discovered)

    if sys.executable:
        candidates.append(sys.executable)

    seen = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        key = os.path.normcase(candidate)
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(candidate):
            return candidate

    raise RuntimeError(
        "No worker Python was found. Run install.bat with a Python 3.10+ "
        "path or set NAFNET_WORKER_PYTHON to python.exe."
    )


def _configured_worker_kind():
    if not os.path.isfile(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(config, dict):
        return None
    value = config.get("worker_kind")
    return value if isinstance(value, str) else None


def _env_truthy(name):
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off", ""):
        return False
    return None


def use_rust_worker():
    """Decide whether the plug-in should invoke the Rust worker.

    Mirrors the LaMa plug-in's resolution order:
    1. ``NAFNET_USE_RUST_WORKER`` env var (opt in/out)
    2. ``worker_kind`` in ``nafnet_config.json`` (``rust`` or ``python``)
    3. Default: Rust worker when binary exists, Python otherwise.
    """
    rust_binary = find_rust_worker()
    if rust_binary is None:
        return False

    truthy = _env_truthy("NAFNET_USE_RUST_WORKER")
    if truthy is True:
        return True
    if truthy is False:
        return False

    config_value = _configured_worker_kind()
    if config_value is not None:
        normalized = config_value.strip().lower()
        if normalized == "rust":
            return True
        if normalized == "python":
            return False

    return True


def find_rust_worker():
    if os.path.isfile(RUST_WORKER_BINARY):
        return RUST_WORKER_BINARY
    return None


def _run_subprocess(command):
    kwargs = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": WORKER_TIMEOUT_SECONDS,
    }
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(command, **kwargs)


def _pulse_during_subprocess(command, phase_text="Working..."):
    """Run a subprocess while pulsing the GIMP progress bar.

    A daemon thread calls ``Gimp.progress_pulse()`` and
    ``Gimp.progress_set_text(phase_text)`` every 250 ms. The
    pulse continues until the subprocess completes (or times
    out, or raises).

    This is the **cheap path** for progress feedback: zero
    changes to the worker, no protocol parsing, just a thread
    on the GIMP side. The downside is that the bar moves
    without a real fraction — GIMP shows the "infinite
    progress" animation, which is the right UX for an
    indeterminate task (longer than 2 s, no known total).

    All Gimp.progress calls go through ``_safe_progress`` so a
    thread-safety issue or a non-interactive GIMP context
    degrades to "no pulse" rather than a crash. The daemon
    thread is ``daemon=True`` so it cannot block the plug-in
    from exiting.

    The informative path (per-tile progress) is deferred — it
    requires the Rust worker to emit ``[LAMA_MARKER] phase
    tile <i>/<n>`` on stderr and the GIMP-side to parse them,
    which is a Rust rebuild.
    """
    stop_event = threading.Event()

    def _pulse_loop():
        # ``stop_event.wait(timeout)`` returns True when the
        # event is set, False on timeout. Looping with
        # ``wait`` instead of ``sleep`` makes shutdown
        # immediate when the worker completes.
        while not stop_event.is_set():
            _safe_progress(Gimp.progress_pulse)
            _safe_progress(Gimp.progress_set_text, phase_text)
            if stop_event.wait(0.25):
                break

    pulse_thread = threading.Thread(
        target=_pulse_loop, name="nafnet-progress-pulse", daemon=True,
    )
    pulse_thread.start()
    try:
        return _run_subprocess(command)
    finally:
        stop_event.set()
        # ``join`` with a timeout so a stuck pulse loop can
        # never delay the plug-in past the worker timeout.
        pulse_thread.join(timeout=WORKER_TIMEOUT_SECONDS + 5)


def _process_detail(completed):
    output = (completed.stderr or completed.stdout or "").strip()
    if not output:
        return f"exit code {completed.returncode}"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    detail = lines[-1] if lines else output
    return detail[:500]


def _windows_no_window_popen_kwargs():
    """Build kwargs that hide the spawned worker console on Windows.

    Both ``CREATE_NO_WINDOW`` and a ``STARTUPINFO`` with
    ``STARTF_USESHOWWINDOW``/``SW_HIDE`` are applied because some Python
    builds on Windows expose one flag but not the other. This only
    hides the spawned ML worker console; the terminal used to launch
    GIMP with ``--verbose`` remains visible.
    """
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "bufsize": 0,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name != "nt":
        return kwargs

    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    startupinfo = None
    start_flags = getattr(subprocess, "STARTF_USESHOWWINDOW", None)
    sw_hide = getattr(subprocess, "SW_HIDE", None)
    if start_flags is not None and sw_hide is not None:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags = start_flags
        startupinfo.wShowWindow = sw_hide
    if startupinfo is not None:
        kwargs["startupinfo"] = startupinfo
    return kwargs


def _run_worker_with_progress(command, progress_callback):
    """Run the ML worker with ``Popen`` so the GIMP UI can stay responsive.

    Mirror of the LaMa plug-in's worker launcher. The daemon reader
    thread reads the merged stdout+stderr line-by-line into a
    ``queue.Queue``. The main thread polls ``process.poll()`` every
    ~0.25 s, drives the GIMP progress bar, and enforces the
    ``WORKER_TIMEOUT_SECONDS`` deadline. On timeout/exit the child
    is terminated and then killed if it doesn't exit; the reader
    thread is joined with a brief timeout so we never block the
    GIMP process on a stuck pipe.

    Returns a ``SimpleNamespace`` mirroring
    ``subprocess.CompletedProcess`` (returncode, stdout, stderr,
    timed_out) so the call sites can share the existing error
    path.
    """
    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    popen_kwargs = _windows_no_window_popen_kwargs()
    process = subprocess.Popen(command, **popen_kwargs)

    output_queue: "queue.Queue[object]" = queue.Queue()

    def _reader():
        """Drain the merged pipe line-by-line into ``output_queue``.

        ``readline`` blocks between lines, which is fine because this
        thread is the only consumer of the pipe and is daemonic. When
        the worker exits (or is terminated/killed) the pipe closes,
        ``readline`` returns ``""``, the iterator terminates, and the
        ``finally`` block closes the pipe handle and signals completion
        with a ``None`` sentinel.
        """
        if process.stdout is None:
            output_queue.put(None)
            return
        try:
            for line in iter(process.stdout.readline, ""):
                output_queue.put(line)
        except Exception:
            pass
        finally:
            try:
                process.stdout.close()
            except Exception:
                pass
            output_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    timed_out = False
    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                break

            if progress_callback is not None:
                try:
                    progress_callback()
                except Exception:
                    pass

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break

            time.sleep(min(WORKER_POLL_INTERVAL_SECONDS, remaining))
    finally:
        if process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
        reader_thread.join(timeout=2.0)

    output_chunks = []
    while True:
        try:
            item = output_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            break
        output_chunks.append(item)

    return SimpleNamespace(
        returncode=process.returncode,
        stdout="".join(output_chunks),
        stderr=None,
        timed_out=timed_out,
    )


def save_buffer_as_png(buffer, path):
    """Save a GEGL buffer through buffer-source -> png-save."""
    graph = Gegl.Node()
    source = graph.create_child("gegl:buffer-source")
    source.set_property("buffer", buffer)
    saver = graph.create_child("gegl:png-save")
    saver.set_property("path", path)
    source.link(saver)
    saver.process()
    if not os.path.isfile(path):
        raise OSError(f"GEGL did not create {path}")


def load_png_into_shadow(drawable, path, expected_width, expected_height):
    """Load a PNG through png-load -> write-buffer into the shadow buffer.

    Used by the whole-image mode: the result is the same size as
    the drawable, so a direct load+write at origin (0, 0) is
    correct.
    """
    shadow_buffer = drawable.get_shadow_buffer()
    graph = Gegl.Node()
    loader = graph.create_child("gegl:png-load")
    loader.set_property("path", path)

    bounds = loader.get_bounding_box()
    if bounds.width != expected_width or bounds.height != expected_height:
        raise ValueError(
            "worker result dimensions differ: "
            f"{bounds.width}x{bounds.height} vs "
            f"{expected_width}x{expected_height}"
        )

    writer = graph.create_child("gegl:write-buffer")
    writer.set_property("buffer", shadow_buffer)
    loader.link(writer)
    writer.process()
    shadow_buffer.flush()


def paste_roi_into_shadow(
    shadow_buffer,
    path,
    sel_x,
    sel_y,
    sel_w,
    sel_h,
    context_px,
    roi_w,
    roi_h,
):
    """Load the worker's result PNG and paste the inner selection bbox
    into the shadow buffer at the right pixel position.

    The worker output is ``roi_w x roi_h`` (selection bbox + the
    64 px context ring, clipped to the image bounds). The user
    only wants the inner ``sel_w x sel_h`` (the actual selection
    bbox, no context ring). This function:

    1. Loads the result PNG.
    2. Crops to the inner selection bbox (offset (context_px,
       context_px), extent (sel_w, sel_h)) via ``gegl:crop``.
       The crop's output extent is at (context_px, context_px)
       in the loaded image's coordinate system (per
       ``gegl_crop_get_bounding_box`` in operations/core/crop.c).
    3. Translates by (sel_x - context_px, sel_y - context_px)
       so the output rectangle lands at (sel_x, sel_y) in the
       drawable's coordinate system.
    4. Writes to the shadow buffer; the input's extent is
       honored so only the inner selection bbox is touched.
       Pixels outside the selection are untouched.

    Without the crop, the user would see a 64 px "halo" of
    restored pixels around their selection (the context ring)
    plus the result would be written at (0, 0) instead of
    (sel_x, sel_y) -- a double bug. The translate fixes the
    position; the crop drops the context ring.

    **Why ``gegl:crop`` (not ``gegl:rectangle``):** the
    previous version used ``gegl:rectangle`` (a render op that
    draws a colored rectangle on top of the input) thinking it
    was a crop. Per ``operations/common/rectangle.c``, the
    rectangle's output comes from an internal
    ``gegl:color -> gegl:crop`` chain, so the input pad is
    effectively ignored -- the result is a green rectangle of
    the requested size, regardless of the loaded PNG's
    content. The user saw "no meaningful restoration" because
    a green rectangle was being pasted into the selection.
    Using ``gegl:crop`` directly fixes this: the crop's input
    is the loaded PNG and the output is the cropped content.

    **Size check fix:** the previous version raised an error
    when ``roi_w != sel_w + 2*context_px`` -- but ``roi_w`` is
    already clipped to the image bounds in the caller, so on
    edge selections it doesn't equal the unclipped ideal.
    The check now compares the loaded PNG's dimensions
    against the caller's ``roi_w, roi_h`` (the actual
    expected size).
    """
    graph = Gegl.Node()
    loader = graph.create_child("gegl:png-load")
    loader.set_property("path", path)

    # Verify the loaded PNG matches the caller's ROI dimensions.
    bounds = loader.get_bounding_box()
    if bounds.width != roi_w or bounds.height != roi_h:
        raise ValueError(
            f"result dimensions ({bounds.width}x{bounds.height}) "
            f"don't match expected ROI size ({roi_w}x{roi_h})"
        )

    crop = graph.create_child("gegl:crop")
    crop.set_property("x", float(context_px))
    crop.set_property("y", float(context_px))
    crop.set_property("width", float(sel_w))
    crop.set_property("height", float(sel_h))
    crop.link(loader)

    translate = graph.create_child("gegl:translate")
    translate.set_property("x", float(sel_x - context_px))
    translate.set_property("y", float(sel_y - context_px))
    translate.link(crop)

    writer = graph.create_child("gegl:write-buffer")
    writer.set_property("buffer", shadow_buffer)
    writer.link(translate)

    graph.process()
    shadow_buffer.flush()


def _diagnose_gegl_buffer(buffer):
    """Log the buffer's extent and (if available) its format.

    These are safe reads that work at runtime (after Gimp.main has
    set up the wire protocol). They are intentionally NOT used to
    branch on behaviour; they only produce diagnostic lines in
    nafnet.log so the next failure shows the exact buffer state.
    """
    try:
        extent = buffer.get_extent()
        _log("diag: buffer extent x=" + str(extent.x) + " y=" + str(extent.y)
             + " w=" + str(extent.width) + " h=" + str(extent.height))
    except Exception as exc:
        _log("diag: buffer.get_extent() failed: " + type(exc).__name__ + ": " + str(exc))
    try:
        fmt = buffer.get_property("format")
        _log("diag: buffer format=" + str(fmt))
    except Exception:
        # gegl:buffer does not necessarily expose format via get_property.
        # Fall back to babl_format() through the C API; if that's not
        # available either, we just don't log it.
        try:
            import ctypes
            _log("diag: buffer format introspection not available in this PyGObject build")
        except Exception:
            pass


def _diagnose_gegl_node(node, label):
    """Log the operation's class name and the list of declared
    properties. Used to confirm a node has the properties the
    plug-in is about to set/get on it. Safe at runtime (pure reads).
    """
    try:
        op = node.get_property("gegl:operation-name")
        _log("diag: " + label + " op=" + str(op))
    except Exception as exc:
        _log("diag: " + label + " get gegl:operation-name failed: "
             + type(exc).__name__ + ": " + str(exc))
    try:
        op_class = node.get_operation()
        if op_class is not None:
            gtype = op_class.__gtype__
            _log("diag: " + label + " gtype=" + str(gtype))
            # List declared properties via the GObject interface
            # (the GParamSpec table). This is what fails on
            # non-existent properties.
            try:
                from gi.repository import GObject
                for pspec in GObject.type_class_list_properties(gtype):
                    _log("diag: " + label + " property: "
                         + pspec.name + " (value_type=" + str(pspec.value_type) + ")")
            except Exception as exc:
                _log("diag: " + label + " list_properties failed: "
                     + type(exc).__name__ + ": " + str(exc))
        else:
            _log("diag: " + label + " no operation (raw GeglNode)")
    except Exception as exc:
        _log("diag: " + label + " get_operation failed: "
             + type(exc).__name__ + ": " + str(exc))


def extract_alpha_png(src_buffer, path):
    """Save the alpha channel of an RGBA buffer as a Y u8 PNG.

    Used to preserve the original alpha through the sidecar: the
    GIMP side extracts the alpha, the worker combines it with the
    model's RGB output, the GIMP side loads the combined RGBA back.

    Pipeline (verified against GEGL 0.4.70):
        buffer-source --(output pad)--> component-extract
        component-extract --(output pad)--> png-save
    The previous version tried to call `extract.get_property("buffer")`,
    but `gegl:component-extract` does not declare a `buffer` property
    (its only properties are `component`, `invert`, `linear` per
    operations/common/component-extract.c). The GValue was left at
    G_TYPE_INVALID and PyGObject's _pygi_value_to_pyobject raised
    TypeError("Invalid type") at pygi-value.c:782. The fix is to
    link the extract node's output pad directly to png-save's
    input pad and call saver.process() at the end.

    The caller MUST only invoke this when the drawable actually has
    an alpha channel (use `drawable.has_alpha()`). For RGB / GRAY
    drawables the caller should pass `alpha_path=None` to the
    worker, which will fall back to the input PNG's alpha (all 255
    for RGB images).
    """
    _log("extract_alpha_png: start (path=" + path + ")")
    _diagnose_gegl_buffer(src_buffer)

    graph = Gegl.Node()
    _log("extract_alpha_png: creating buffer-source")
    source = graph.create_child("gegl:buffer-source")
    _diagnose_gegl_node(source, "buffer-source")
    _log("extract_alpha_png: set source.buffer")
    source.set_property("buffer", src_buffer)

    _log("extract_alpha_png: creating component-extract")
    extract = graph.create_child("gegl:component-extract")
    _diagnose_gegl_node(extract, "component-extract")
    _log("extract_alpha_png: set extract.component=alpha")
    extract.set_property("component", "alpha")

    _log("extract_alpha_png: creating png-save")
    saver = graph.create_child("gegl:png-save")
    _diagnose_gegl_node(saver, "png-save")
    _log("extract_alpha_png: set saver.path=" + path)
    saver.set_property("path", path)

    _log("extract_alpha_png: linking source -> extract")
    source.link(extract)
    _log("extract_alpha_png: linking extract -> saver")
    extract.link(saver)

    _log("extract_alpha_png: saver.process() (runs the whole chain)")
    saver.process()
    _log("extract_alpha_png: saver.process() returned")

    if not os.path.isfile(path):
        raise OSError("gegl:png-save did not create " + path)
    _log("extract_alpha_png: done; " + path + " size=" + str(os.path.getsize(path)))


def _extract_alpha_from_png(image_path, alpha_path):
    """Extract the alpha channel of a PNG file and save it as Y u8 PNG.

    Same GEGL pattern as ``extract_alpha_png`` (buffer-source ->
    component-extract -> png-save) but starting from a file path
    rather than a GeglBuffer. The png-load node reads the input
    PNG, the component-extract node pulls out the alpha, the
    png-save node writes it. ``saver.process()`` at the end runs
    the whole chain.

    Used by the region code path: the cropped region is first
    saved as RGB(A) PNG (via the rectangle pipeline), then
    re-loaded through png-load to feed the alpha-extract chain.
    The intermediate PNG load is a few hundred KB and adds
    <50 ms; we accept the cost in exchange for a pipeline that
    doesn't depend on a non-existent buffer property.
    """
    graph = Gegl.Node()
    loader = graph.create_child("gegl:png-load")
    loader.set_property("path", image_path)
    extract = graph.create_child("gegl:component-extract")
    extract.set_property("component", "alpha")
    loader.link(extract)
    saver = graph.create_child("gegl:png-save")
    saver.set_property("path", alpha_path)
    extract.link(saver)
    saver.process()
    if not os.path.isfile(alpha_path):
        raise OSError("gegl:png-save did not create " + alpha_path)


def _return_error(procedure, status, message):
    error = GLib.Error.new_literal(Gimp.PlugIn.error_quark(), message, 0)
    return procedure.new_return_values(status, error)


def _calling_error(procedure, message):
    return _return_error(procedure, Gimp.PDBStatusType.CALLING_ERROR, message)


def _execution_error(procedure, message):
    return _return_error(procedure, Gimp.PDBStatusType.EXECUTION_ERROR, message)


def _safe_progress(callable_, *args, **kwargs):
    """Invoke a Gimp.progress_* function without aborting on failure.

    GIMP can be in non-interactive contexts (batch, transitions)
    where the progress callbacks are no-ops; the wrapper keeps the
    plug-in functional in those states.
    """
    try:
        return callable_(*args, **kwargs)
    except Exception:
        return None


# Worker-kind decision must run after the helpers that depend on
# queue / subprocess are defined. Inlined here to keep the
# top-of-file imports lean.


class NafnetRestore(Gimp.PlugIn):
    def do_set_i18n(self, _name):
        return False

    def do_query_procedures(self):
        return [
            "plug-in-nafnet-restore",
            "plug-in-nafnet-restore-region",
        ]

    def _create_procedure(self, name, menu_label, blurb):
        Gegl.init(None)
        procedure = Gimp.ImageProcedure.new(
            self,
            name,
            Gimp.PDBProcType.PLUGIN,
            self.run,
            None,
        )
        procedure.set_image_types("RGB*, GRAY*")
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        procedure.set_menu_label(menu_label)
        procedure.set_icon_name(GimpUi.ICON_GEGL)
        procedure.add_menu_path("<Image>/Filters/Enhance/")
        procedure.set_documentation(
            blurb,
            "Exports the drawable (+ optional selection) to a sidecar "
            "worker, runs NAFNet image restoration, then loads the "
            "result back into the drawable's shadow buffer.",
            name,
        )
        procedure.set_attribution(
            "GIMP Inpainting Plug-in",
            "GIMP Inpainting Plug-in",
            "2026",
        )
        return procedure

    def do_create_procedure(self, name):
        if name == "plug-in-nafnet-restore":
            return self._create_procedure(
                name,
                "_Restore Image (NAFNet)...",
                "Run NAFNet image restoration on the entire active "
                "drawable. ~9 s per 1024x1024 tile. Best for general "
                "deblurring and restoration of natural photographs.",
            )
        if name == "plug-in-nafnet-restore-region":
            return self._create_procedure(
                name,
                "_Restore Selection (NAFNet)...",
                "Run NAFNet image restoration only inside the active "
                "selection. A " + str(SELECTION_CONTEXT_PX) + " px context "
                "ring is added around the selection bbox so the model "
                "has surrounding pixels to inform the restoration. "
                "Pixels outside the original selection are not modified.",
            )
        return None

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        name = procedure.get_name()
        _log(
            "run: name=" + repr(name)
            + " run_mode=" + repr(run_mode)
            + " image=" + repr(image)
            + " drawables=" + repr(len(drawables) if drawables else 0)
        )
        try:
            if name == "plug-in-nafnet-restore-region":
                return self._run_region(procedure, run_mode, image, drawables)
            return self._run_whole(procedure, run_mode, image, drawables)
        except Exception as exc:
            import traceback as _tb
            _log("run: UNCAUGHT EXCEPTION " + type(exc).__name__ + ": " + str(exc))
            for line in _tb.format_exc().splitlines():
                _log("  | " + line)
            return _execution_error(
                procedure, "NAFNet plug-in crashed: " + str(exc),
            )

    # ----------------- Worker backend -----------------

    def _resolve_worker_command(self, image_path, output_path, alpha_path=None):
        """Build the command line for the selected worker kind.

        Rust worker takes the same CLI flags as the Python worker.
        Both write a PNG to ``output_path`` and emit
        ``[LAMA_MARKER] phase <name>`` lines on stderr for progress
        parsing.

        ``alpha_path`` is optional: when provided, the worker reads
        the alpha channel from it and combines the model's RGB
        output with the original alpha before writing. This
        preserves alpha through the sidecar round-trip for RGBA
        drawables.
        """
        rust_binary = find_rust_worker()
        use_rust = use_rust_worker() and rust_binary is not None

        if use_rust:
            cmd = [rust_binary, "--image", image_path, "--output", output_path, "--model", MODEL_PATH]
            if alpha_path is not None:
                cmd += ["--alpha", alpha_path]
            return cmd, True
        if not os.path.isfile(WORKER_SCRIPT):
            raise FileNotFoundError(f"Worker script is missing: {WORKER_SCRIPT}. Reinstall the plug-in.")

        worker_python = find_worker_python()
        # We don't probe onnxruntime / numpy / Pillow here: if the
        # worker script's import fails, the worker process exits with
        # a non-zero code and the call site surfaces the detail in
        # the GIMP error dialog.
        cmd = [worker_python, WORKER_SCRIPT, "--image", image_path, "--output", output_path, "--model", MODEL_PATH]
        if alpha_path is not None:
            cmd += ["--alpha", alpha_path]
        return cmd, False

    def _run_whole(self, procedure, run_mode, image, drawables):
        """Restore the entire active drawable.

        No selection interaction: the model processes the full
        drawable. Use for whole-image deblurring.
        """
        progress_started = False

        def _phase(text, fraction):
            if not progress_started:
                return
            _safe_progress(Gimp.progress_set_text, text)
            _safe_progress(Gimp.progress_update, fraction)

        def _pulse():
            if not progress_started:
                return
            _safe_progress(Gimp.progress_pulse)
            _safe_progress(
                Gimp.progress_set_text,
                "Running NAFNet inference (NN opaque)... please wait",
            )

        def _worker_progress_callback():
            _pulse()

        try:
            if len(drawables) != 1:
                return _calling_error(
                    procedure,
                    f"NAFNet Restore requires exactly one drawable; got {len(drawables)}.",
                )

            drawable = drawables[0]
            _log("_run_whole: drawable type=" + str(type(drawable).__name__))
            width = drawable.get_width()
            height = drawable.get_height()
            _log("_run_whole: drawable size=" + str(width) + "x" + str(height))
            if width <= 0 or height <= 0:
                return _calling_error(procedure, "The active drawable is empty.")

            if not os.path.isfile(MODEL_PATH):
                return _calling_error(
                    procedure,
                    f"NAFNet model is missing: {MODEL_PATH}. Reinstall the plug-in.",
                )

            _safe_progress(Gimp.progress_init, "NAFNet Restore")
            progress_started = True
            _phase("Preparing image...", 0.05)

            try:
                with tempfile.TemporaryDirectory(prefix="gimp-nafnet-") as temp_dir:
                    _log("_run_whole: temp_dir=" + temp_dir)
                    image_path = os.path.join(temp_dir, "image.png")
                    has_alpha = drawable.has_alpha()
                    alpha_path = os.path.join(temp_dir, "alpha.png") if has_alpha else None
                    output_path = os.path.join(temp_dir, "result.png")
                    _log("_run_whole: has_alpha=" + str(has_alpha))

                    _log("_run_whole: drawable.get_buffer()")
                    full_buffer = drawable.get_buffer()
                    _log("_run_whole: save_buffer_as_png(image.png)")
                    save_buffer_as_png(full_buffer, image_path)
                    _log("_run_whole: image.png saved")

                    if has_alpha:
                        _log("_run_whole: extracting alpha.png")
                        extract_alpha_png(full_buffer, alpha_path)
                        _log("_run_whole: alpha.png extracted")
                    else:
                        _log("_run_whole: no alpha to extract; worker will use input alpha")

                    _phase("Running NAFNet worker...", 0.20)
                    # Worker selection. The Rust worker (default) has
                    # 512x512 tiled inference + 2D tent blend, which
                    # avoids the BFC arena OOM that the single-pass
                    # Python worker hits on images >~1 Mpix. The Python
                    # worker remains the fallback.
                    _log("_run_whole: resolving worker command")
                    try:
                        command, use_rust = self._resolve_worker_command(
                            image_path, output_path, alpha_path=alpha_path,
                        )
                    except FileNotFoundError as exc:
                        _log("_run_whole: worker FileNotFoundError: " + str(exc))
                        return _execution_error(procedure, str(exc))
                    except RuntimeError as exc:
                        _log("_run_whole: worker RuntimeError: " + str(exc))
                        return _calling_error(procedure, str(exc))

                    _log("_run_whole: command=" + " ".join(command))
                    if use_rust:
                        _log("worker: rust (tiled)")
                    else:
                        _log("worker: python (single-pass; may OOM on >1 Mpix)")

                    _log("_run_whole: spawning worker (with progress pulse)")
                    completed = _pulse_during_subprocess(
                        command,
                        phase_text="Running NAFNet inference "
                                  "(~30 s for 2K images)...",
                    )
                    _log("_run_whole: worker rc=" + str(completed.returncode))
                    if completed.returncode != 0:
                        _log("_run_whole: worker stderr:\n" + (completed.stderr or "<empty>"))
                        _log("_run_whole: worker stdout:\n" + (completed.stdout or "<empty>"))
                        return _execution_error(
                            procedure,
                            "The NAFNet worker failed: " + _process_detail(completed),
                        )
                    if not os.path.isfile(output_path):
                        _log("_run_whole: worker did not create " + output_path)
                        return _execution_error(
                            procedure,
                            "The NAFNet worker did not create its output PNG.",
                        )

                    _phase("Loading result...", 0.90)
                    _log("_run_whole: load_png_into_shadow")
                    load_png_into_shadow(drawable, output_path, width, height)
                    _log("_run_whole: load_png_into_shadow done")

                # Refresh the entire drawable ΓÇö the whole image was
                # replaced. We do not constrain the update to a
                # region because for whole-image mode every pixel
                # may have changed.
                _log("_run_whole: merge_shadow + update + displays_flush")
                drawable.merge_shadow(True)
                drawable.update(0, 0, width, height)
                Gimp.displays_flush()
                _log("_run_whole: paste + flush done")
            except (OSError, RuntimeError, ValueError) as exc:
                _log("_run_whole: image transfer caught exception: " + type(exc).__name__ + ": " + str(exc))
                return _execution_error(procedure, f"NAFNet image transfer failed: {exc}")
            except Exception as exc:
                import traceback as _tb
                _log("_run_whole: outer caught exception: " + type(exc).__name__ + ": " + str(exc))
                for line in _tb.format_exc().splitlines():
                    _log("  | " + line)
                return _execution_error(procedure, f"NAFNet Restore failed: {exc}")

            _phase("Complete", 1.0)
            return procedure.new_return_values(
                Gimp.PDBStatusType.SUCCESS,
                GLib.Error(),
            )
        finally:
            if progress_started:
                _safe_progress(Gimp.progress_end)

    def _run_region(self, procedure, run_mode, image, drawables):
        """Restore only the selected region.

        The selection bbox is expanded by ``SELECTION_CONTEXT_PX`` on
        every side, the expanded region is cropped, sent to the
        worker, and the result is pasted back into the original
        drawable with the original selection bbox as the destination.
        Pixels outside the original selection bbox are not modified.
        """
        progress_started = False

        def _phase(text, fraction):
            if not progress_started:
                return
            _safe_progress(Gimp.progress_set_text, text)
            _safe_progress(Gimp.progress_update, fraction)

        def _pulse():
            if not progress_started:
                return
            _safe_progress(Gimp.progress_pulse)
            _safe_progress(
                Gimp.progress_set_text,
                "Running NAFNet inference on the selection... please wait",
            )

        def _worker_progress_callback():
            _pulse()

        try:
            if len(drawables) != 1:
                return _calling_error(
                    procedure,
                    f"NAFNet Restore Region requires exactly one drawable; got {len(drawables)}.",
                )

            drawable = drawables[0]
            _log("_run_region: drawable type=" + str(type(drawable).__name__))
            # Detect "no active selection" with Selection.is_empty
            # -- mask_intersect cannot do this, it always returns
            # True with the full drawable bounds when there's no
            # selection (which would silently let a no-selection
            # click fall through and process the entire image).
            is_empty = Gimp.Selection.is_empty(image)
            _log("_run_region: Gimp.Selection.is_empty=" + str(is_empty))
            if is_empty:
                # No active selection. We can't grey out the menu
                # item by selection (GIMP 3.2's sensitivity mask
                # has no SELECTION flag, and do_set_sensitivity
                # runs before the user clicks so the image
                # argument is None). The best we can do is warn
                # via Gimp.message -- that shows briefly in the
                # GIMP status bar AND logs to the Error Console
                # (Windows > Dockable Dialogs > Error Console).
                # We return early without raising an error so no
                # dialog blocks the workflow.
                Gimp.message(
                    "NAFNet Restore Region: no selection detected. "
                    "Use the Rectangle Select tool (R) to make a "
                    "non-empty selection, then try again. "
                    "(GIMP 3.2 has no API to grey out menu items "
                    "based on selection state; this warning is "
                    "logged to the Error Console.)"
                )
                _log("_run_region: no selection; Gimp.message warning emitted; returning early")
                return procedure.new_return_values(
                    Gimp.PDBStatusType.SUCCESS, GLib.Error(),
                )

            # Now that we know there's a selection, get its
            # bounds. mask_intersect returns
            # (intersects, x, y, width, height) clipped to the
            # drawable -- the same values the rest of the
            # pipeline needs (sel_x, sel_y, sel_w, sel_h).
            intersects, sel_x, sel_y, sel_w, sel_h = drawable.mask_intersect()
            _log("_run_region: mask_intersect=" + repr((intersects, sel_x, sel_y, sel_w, sel_h)))

            width = drawable.get_width()
            height = drawable.get_height()
            _log("_run_region: drawable size=" + str(width) + "x" + str(height))
            if width <= 0 or height <= 0:
                return _calling_error(procedure, "The active drawable is empty.")

            if not os.path.isfile(MODEL_PATH):
                return _calling_error(
                    procedure,
                    f"NAFNet model is missing: {MODEL_PATH}. Reinstall the plug-in.",
                )

            # Compute the expanded ROI in original-image pixel
            # coordinates. The bbox+context is the region the model
            # processes; only the original selection bbox is pasted
            # back, so the context is just for the model.
            ctx = SELECTION_CONTEXT_PX
            roi_x = max(0, sel_x - ctx)
            roi_y = max(0, sel_y - ctx)
            roi_w = min(width, sel_x + sel_w + ctx) - roi_x
            roi_h = min(height, sel_y + sel_h + ctx) - roi_y

            _log(f"region-restore: selection {sel_x},{sel_y} {sel_w}x{sel_h}, "
                 f"ROI {roi_x},{roi_y} {roi_w}x{roi_h}")

            _safe_progress(Gimp.progress_init, "NAFNet Restore Region")
            progress_started = True
            _phase("Preparing selection...", 0.05)

            try:
                with tempfile.TemporaryDirectory(prefix="gimp-nafnet-region-") as temp_dir:
                    _log("_run_region: temp_dir=" + temp_dir)
                    image_path = os.path.join(temp_dir, "region.png")
                    has_alpha = drawable.has_alpha()
                    alpha_path = os.path.join(temp_dir, "alpha.png") if has_alpha else None
                    output_path = os.path.join(temp_dir, "result.png")

                    _phase("Cropping to selection + context...", 0.10)
                    # Crop the drawable's shadow buffer to the ROI
                    # and save. Pipeline: buffer-source -> gegl:crop
                    # -> png-save, with saver.process() at the end.
                    # We use ``gegl:crop`` directly (NOT
                    # ``gegl:rectangle``, which is a render op that
                    # draws a green rectangle of the requested size
                    # regardless of the input -- the previous version
                    # used it here and the worker ended up processing
                    # a green rectangle, so the user saw no visible
                    # difference between Restore Image and Restore
                    # Selection when the selection was the whole
                    # image). Same root cause as the alpha and
                    # paste-pipeline bugs.
                    _log("_run_region: getting shadow buffer (has_alpha=" + str(has_alpha) + ")")
                    full_shadow = drawable.get_shadow_buffer()
                    _log("_run_region: shadow buffer obtained")
                    graph = Gegl.Node()
                    source = graph.create_child("gegl:buffer-source")
                    source.set_property("buffer", full_shadow)
                    crop = graph.create_child("gegl:crop")
                    crop.set_property("x", float(roi_x))
                    crop.set_property("y", float(roi_y))
                    crop.set_property("width", float(roi_w))
                    crop.set_property("height", float(roi_h))
                    _log("_run_region: linking source -> crop")
                    source.link(crop)
                    saver = graph.create_child("gegl:png-save")
                    saver.set_property("path", image_path)
                    _log("_run_region: linking crop -> saver")
                    crop.link(saver)
                    _log("_run_region: saver.process() (runs the whole chain)")
                    saver.process()
                    if not os.path.isfile(image_path):
                        raise OSError("gegl:png-save did not create " + image_path)
                    _log("_run_region: region.png saved size=" + str(os.path.getsize(image_path)))
                    if has_alpha:
                        _log("_run_region: extracting alpha from region.png")
                        # Re-load region.png through png-load to
                        # extract alpha. We don't have a GeglBuffer
                        # in hand (the previous version's broken
                        # pipeline left us without one), so the
                        # alpha extraction is its own GEGL graph:
                        # png-load -> component-extract -> png-save.
                        _extract_alpha_from_png(image_path, alpha_path)
                        _log("_run_region: alpha.png extracted")
                    else:
                        _log("_run_region: no alpha to extract (RGB drawable); worker will use input alpha")

                    _phase("Starting NAFNet worker...", 0.20)
                    _log("_run_region: resolving worker command")
                    try:
                        command, use_rust = self._resolve_worker_command(
                            image_path, output_path, alpha_path=alpha_path,
                        )
                    except FileNotFoundError as exc:
                        _log("_run_region: worker FileNotFoundError: " + str(exc))
                        return _execution_error(procedure, str(exc))
                    except RuntimeError as exc:
                        _log("_run_region: worker RuntimeError: " + str(exc))
                        return _calling_error(procedure, str(exc))

                    _log("_run_region: command=" + " ".join(command))
                    if use_rust:
                        _log("worker: rust")
                    else:
                        _log("worker: python")

                    try:
                        _log("_run_region: spawning worker")
                        completed = _run_worker_with_progress(
                            command, _worker_progress_callback
                        )
                    except OSError as exc:
                        _log("_run_region: spawn OSError: " + str(exc))
                        return _execution_error(
                            procedure,
                            f"Could not start the NAFNet worker: {exc}",
                        )

                    _log("_run_region: worker exited rc=" + str(completed.returncode)
                         + " timed_out=" + str(completed.timed_out))
                    if completed.timed_out:
                        return _execution_error(
                            procedure,
                            f"The NAFNet worker timed out after "
                            f"{WORKER_TIMEOUT_SECONDS} seconds.",
                        )
                    if completed.returncode != 0:
                        _log("_run_region: worker stderr:\n" + (completed.stderr or "<empty>"))
                        return _execution_error(
                            procedure,
                            "The NAFNet worker failed: "
                            + _process_detail(completed),
                        )
                    if not os.path.isfile(output_path):
                        _log("_run_region: worker did not create " + output_path)
                        return _execution_error(
                            procedure,
                            "The NAFNet worker did not create its output PNG.",
                        )

                    _phase("Pasting result back...", 0.90)
                    # Load the result PNG (size roi_w x roi_h,
                    # including context ring) and paste the inner
                    # selection bbox (sel_w x sel_h) into the
                    # shadow buffer at the correct pixel position
                    # (sel_x, sel_y). The crop drops the context ring
                    # so the user only sees the restoration they
                    # selected; the translate fixes the position
                    # (without it, the result would land at (0, 0)
                    # in the drawable, which is a bug).
                    _log("_run_region: getting shadow buffer for paste")
                    shadow = drawable.get_shadow_buffer()
                    try:
                        _log("_run_region: pasting result ROI")
                        paste_roi_into_shadow(
                            shadow_buffer=shadow,
                            path=output_path,
                            sel_x=sel_x,
                            sel_y=sel_y,
                            sel_w=sel_w,
                            sel_h=sel_h,
                            context_px=SELECTION_CONTEXT_PX,
                            roi_w=roi_w,
                            roi_h=roi_h,
                        )
                        _log("_run_region: paste complete")
                    except ValueError as exc:
                        _log("_run_region: paste ValueError: " + str(exc))
                        return _execution_error(procedure, str(exc))

                _log("_run_region: merge_shadow + update + displays_flush")
                drawable.merge_shadow(True)
                # Update only the original selection bbox ΓÇö the
                # GIMP display refresh is bounded to what the user
                # actually selected. The result inside the
                # selection is the restored RGB; pixels outside are
                # untouched.
                drawable.update(sel_x, sel_y, sel_w, sel_h)
                Gimp.displays_flush()
                _log("_run_region: paste + flush done")
            except (OSError, RuntimeError, ValueError) as exc:
                _log("_run_region: image transfer caught exception: " + type(exc).__name__ + ": " + str(exc))
                return _execution_error(procedure, f"NAFNet region image transfer failed: {exc}")
            except Exception as exc:
                return _execution_error(procedure, f"NAFNet Restore Region failed: {exc}")

            _phase("Complete", 1.0)
            return procedure.new_return_values(
                Gimp.PDBStatusType.SUCCESS,
                GLib.Error(),
            )
        finally:
            if progress_started:
                _safe_progress(Gimp.progress_end)
Gimp.main(NafnetRestore.__gtype__, sys.argv)
