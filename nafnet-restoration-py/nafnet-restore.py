#!nafnet-gimp-python
"""GIMP 3.x plug-in for NAFNet image restoration.

Exposes two image-scoped procedures that share this single shebang alias:

- ``plug-in-nafnet-restore`` — restore the entire active drawable.
- ``plug-in-nafnet-restore-region`` — restore only the selected
  region (uses the selection bbox with a small context padding so
  the model has surrounding pixels to inform the restoration).

Both spawn a sidecar worker (Rust by default, Python fallback) that
runs NAFNet-REDS on the RGB channels and writes a PNG; the plug-in
loads the PNG back into the drawable's shadow buffer and merges.

Architecture mirrors the LaMa plug-in:

  GIMP process  (MINGW Python 3.14, gi + GEGL only)
    └─ #!nafnet-gimp-python  (shebang → user-level .interp mapping)
         └─ nafnet-restore.py
              ├─ exports drawable (+ optional selection mask) to temp PNGs
              ├─ spawns nafnet_worker_rust.exe (default) or nafnet_worker.py
              └─ loads result PNG into drawable shadow buffer, merge_shadow
                        │
                        ▼
             worker process  (Rust or Python, with Pillow + onnxruntime)
                 ort CPU provider
                 ?? model pipeline: load PNG -> HWC f32 [0, 1] -> (Rust:
                    tile + blend overlap) -> NAFNet -> HWC f32 -> save PNG

NAFNet is 1:1 spatial (input H,W == output H,W) and 3-channel RGB
(red, green, blue only). Alpha is dropped on the way in and not
restored; if the user wants alpha preservation, the GIMP-side
plug-in can copy it back from the original drawable after applying
the restoration. (TBD: not implemented in v1.)
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


# --- diagnostic infrastructure -----------------------------------------
# Write a timestamped line to sys.stderr with explicit flush.
# GIMP captures the plug-in's stderr (via the wire protocol) and
# surfaces it in the *Error Console* (Windows > Error Console)
# when stderr is a TTY. From a non-TTY environment (e.g., user
# launches GIMP from a terminal), stderr lines appear in real
# time on the terminal. The ``flush=True`` is critical: GIMP may
# kill the plug-in child before its buffers are flushed, in which
# case unflushed output is lost.
import os as _os_for_pid
import sys as _sys_for_stderr
import time as _t_for_stderr


def _stderr(msg):
    ts = _t_for_stderr.strftime("%H:%M:%S")
    # Use the stderr FileObject directly. The ``sys`` module
    # itself doesn't have a ``.write`` method — only the ``sys.stderr``
    # and ``sys.stdout`` FileObjects do. ``flush=True`` is required
    # because GIMP may kill the plug-in child before its buffers
    # are flushed, in which case the diagnostic output is lost.
    _sys_for_stderr.stderr.write(f"[nafnet {ts}] {msg}\n")
    _sys_for_stderr.stderr.flush()


# Module-load marker. If the user launches GIMP from a terminal
# and the plug-in file is being loaded, this line should appear
# immediately. If it doesn't, the file is not being executed at
# all (likely a wrong shebang / .interp mapping, or the plug-in
# is in the wrong directory).
_stderr(f"module loaded: nafnet-restore.py (pid={_os_for_pid.getpid()})")


# Stale .pyc detection. If a `__pycache__` directory next to this
# file holds a .pyc compiled by a different Python version (e.g.,
# Python 3.13 left behind by running the test pipeline, while GIMP
# uses Python 3.14), the import silently fails. Detect that and
# print a clear instruction to stderr. The Gimp-side plug-in manager
# catches stderr at scan time, so the message reaches the Error
# Console and the terminal.
import sys as _sys_version_check
_py_version = "%d%d" % _sys_version_check.version_info[:2]
_pycache_dir = _os_for_pid.path.join(_os_for_pid.path.dirname(__file__), "__pycache__")
if _os_for_pid.path.isdir(_pycache_dir):
    for _pyc_name in _os_for_pid.listdir(_pycache_dir):
        if _pyc_name.endswith(".pyc") and not _pyc_name.startswith(
            _os_for_pid.path.basename(__file__).rsplit(".", 1)[0]
            + ".cpython-" + _py_version
        ):
            _stderr(
                f"WARNING: stale bytecode detected: "
                f"{_pycache_dir}\\{_pyc_name}\n"
                f"  This .pyc was not compiled for the current Python "
                f"(expected cpython-{_py_version}). If GIMP silently "
                f"fails to register the plug-in after a code change, "
                f"delete the __pycache__ directory and restart GIMP:\n"
                f"    Remove-Item -Recurse -Force {_pycache_dir}"
            )
            break
# No `del` needed: these are local function-level names, Python's
# reference counting cleans them up when this scope exits. (Adding a
# `del` here would crash with NameError if the for-loop above had
# zero iterations, because `_pyc_name` would never have been bound.)


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

    The worker output is `roi_w x roi_h` (selection bbox + context
    ring). The user only wants the inner `sel_w x sel_h` (the actual
    selection bbox, no context ring). This function:

    1. Loads the result PNG (default origin 0, 0).
    2. Crops to the inner selection bbox (offset (context_px,
       context_px), extent (sel_w, sel_h)). The crop's output
       rectangle is at (context_px, context_px) in the input's
       coordinate system.
    3. Translates by (sel_x - context_px, sel_y - context_px) so
       the output rectangle lands at (sel_x, sel_y) in the
       drawable's coordinate system.
    4. Writes to the shadow buffer at origin (sel_x, sel_y) —
       only the inner selection bbox is touched, the rest of the
       shadow buffer is untouched.

    Without the crop, the user would see a 64 px "halo" of restored
    pixels around their selection (the context ring) plus the
    result would be written at (0, 0) instead of (sel_x, sel_y) — a
    double bug. The translate fixes the position; the rectangle
    drops the context ring.
    """
    expected_w = sel_w + 2 * context_px
    expected_h = sel_h + 2 * context_px
    if roi_w != expected_w or roi_h != expected_h:
        raise ValueError(
            f"result dimensions ({roi_w}x{roi_h}) don't match expected "
            f"ROI+context size ({expected_w}x{expected_h})"
        )

    graph = Gegl.Node()
    loader = graph.create_child("gegl:png-load")
    loader.set_property("path", path)

    crop = graph.create_child("gegl:rectangle")
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


def extract_alpha_png(src_buffer, path):
    """Save the alpha channel of an RGBA buffer as a Y u8 PNG.

    Used to preserve the original alpha through the sidecar: the
    GIMP side extracts the alpha, the worker combines it with the
    model's RGB output, and the GIMP side loads the combined RGBA back.

    The extracted buffer is a single-channel Y u8. `save_buffer_as_png`
    saves whatever format the buffer is in, so this produces a
    grayscale PNG.

    **Caller must verify the source buffer has an alpha channel**
    first (via `_drawable_has_alpha`). Calling this on an RGB buffer
    fails with \"Invalid type\" because the GEGL buffer has no
    alpha component to extract. This is the bug that surfaced in
    the first round of testing.
    """
    graph = Gegl.Node()
    source = graph.create_child("gegl:buffer-source")
    source.set_property("buffer", src_buffer)
    extract = graph.create_child("gegl:component-extract")
    extract.set_property("component", "alpha")
    extract.link(source)
    extract.process()
    save_buffer_as_png(extract.get_property("buffer"), path)


def _drawable_has_alpha(drawable):
    """Return True if the active drawable's image has an alpha channel.

    Looks at the image's base type (Gimp.ImageBaseType) rather than
    the drawable's type. The base type tells us the image's color
    space: RGB / GRAY / INDEXED have no alpha; RGBA / GRAYA /
    INDEXEDA do. For our 1:1 image-restoration use case, this is
    the right check — it tells us whether the alpha round-trip is
    meaningful for the image the user is restoring.

    Implementation notes:
    - C API equivalent: ``gimp_image_get_base_type()`` in
      ``libgimp/gimpimage_pdb.h`` (returns ``GimpImageBaseType``).
    - The drawable-level ``gimp_drawable_type_with_alpha()`` is
      also available, but it forces RGBA on indexed drawables which
      isn't what we want. The image's base type is the right
      property to check.
    """
    image = drawable.get_image()
    base_type = image.get_base_type()
    return base_type in (
        Gimp.ImageBaseType.RGBA,
        Gimp.ImageBaseType.GRAYA,
        Gimp.ImageBaseType.INDEXEDA,
    )


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
        _stderr("do_set_i18n called")
        return False

    def do_query_procedures(self):
        procs = [
            "plug-in-nafnet-restore",
            "plug-in-nafnet-restore-region",
        ]
        _stderr(f"do_query_procedures -> {procs}")
        return procs

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
                "selection. A small context padding is added around "
                "the selection bbox so the model has surrounding "
                "pixels to inform the restoration. Pixels outside the "
                "original selection are not modified.",
            )
        return None

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        try:
            name = procedure.get_name()
            n = len(drawables) if drawables is not None else 0
            _stderr(f"run() called: name={name!r} drawables={n}")
        except Exception as exc:
            _stderr(f"run() entry failed: {type(exc).__name__}: {exc}")
            raise
        if name == "plug-in-nafnet-restore-region":
            return self._run_region(procedure, run_mode, image, drawables)
        return self._run_whole(procedure, run_mode, image, drawables)

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
        _stderr("_run_whole entered")
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
            width = drawable.get_width()
            height = drawable.get_height()
            if width <= 0 or height <= 0:
                return _calling_error(procedure, "The active drawable is empty.")

            if not os.path.isfile(MODEL_PATH):
                return _calling_error(
                    procedure,
                    f"NAFNet model is missing: {MODEL_PATH}. Reinstall the plug-in.",
                )

            has_alpha = _drawable_has_alpha(drawable)
            _log(
                f"whole-restore: drawable={width}x{height} "
                f"has_alpha={has_alpha} "
                f"image.get_base_type()={image.get_base_type()!r}"
            )

            _safe_progress(Gimp.progress_init, "NAFNet Restore")
            progress_started = True
            _phase("Preparing image...", 0.05)

            try:
                with tempfile.TemporaryDirectory(prefix="gimp-nafnet-") as temp_dir:
                    image_path = os.path.join(temp_dir, "image.png")
                    output_path = os.path.join(temp_dir, "result.png")

                    _phase("Preparing image...", 0.10)
                    # Save the drawable buffer. The buffer is
                    # whatever format the drawable uses (RGB or
                    # RGBA). The worker reads only the RGB channels;
                    # the alpha (if any) is preserved separately.
                    full_buffer = drawable.get_buffer()
                    _log(f"whole-restore: buffer.get_format()={full_buffer.get_format()!r}")

                    image_alpha = None
                    if has_alpha:
                        # RGBA image: extract the alpha channel
                        # so the worker can combine it with the
                        # model's RGB output. For RGB images,
                        # there's no alpha component to extract;
                        # calling `gegl:component-extract` on an
                        # RGB buffer fails with "Invalid type".
                        image_alpha = os.path.join(temp_dir, "alpha.png")
                        extract_alpha_png(full_buffer, image_alpha)
                        _log(f"whole-restore: alpha extracted to {image_alpha}")
                    else:
                        _log("whole-restore: no alpha (RGB image), skipping alpha round-trip")

                    _phase("Starting NAFNet worker...", 0.20)
                    try:
                        command, use_rust = self._resolve_worker_command(
                            image_path, output_path, alpha_path=image_alpha,
                        )
                    except FileNotFoundError as exc:
                        return _execution_error(procedure, str(exc))
                    except RuntimeError as exc:
                        return _calling_error(procedure, str(exc))

                    _log(f"whole-restore: command ({len(command)} args): {' '.join(command)}")
                    if use_rust:
                        _log("worker: rust")
                    else:
                        _log("worker: python")

                    try:
                        completed = _run_worker_with_progress(
                            command, _worker_progress_callback
                        )
                    except OSError as exc:
                        return _execution_error(
                            procedure,
                            f"Could not start the NAFNet worker: {exc}",
                        )

                    # Log the worker's full output (marker lines) for
                    # the diagnostic log. On failure, the stderr is
                    # already in completed.stdout (we merged them).
                    if completed.stdout:
                        for line in completed.stdout.splitlines():
                            if line.strip():
                                _log(f"worker: {line.strip()}")

                    if completed.timed_out:
                        return _execution_error(
                            procedure,
                            f"The NAFNet worker timed out after "
                            f"{WORKER_TIMEOUT_SECONDS} seconds.",
                        )
                    if completed.returncode != 0:
                        return _execution_error(
                            procedure,
                            "The NAFNet worker failed: "
                            + _process_detail(completed),
                        )
                    if not os.path.isfile(output_path):
                        return _execution_error(
                            procedure,
                            "The NAFNet worker did not create its output PNG.",
                        )

                    _phase("Applying result...", 0.90)
                    load_png_into_shadow(drawable, output_path, width, height)

                # Refresh the entire drawable — the whole image was
                # replaced. We do not constrain the update to a
                # region because for whole-image mode every pixel
                # may have changed.
                drawable.merge_shadow(True)
                drawable.update(0, 0, width, height)
                Gimp.displays_flush()
            except (OSError, RuntimeError, ValueError) as exc:
                return _execution_error(procedure, f"NAFNet image transfer failed: {exc}")
            except Exception as exc:
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
        _stderr("_run_region entered")
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
            intersects, sel_x, sel_y, sel_w, sel_h = drawable.mask_intersect()
            if not intersects:
                return _calling_error(
                    procedure,
                    "Make a non-empty selection on the active drawable first.",
                )

            width = drawable.get_width()
            height = drawable.get_height()
            if width <= 0 or height <= 0:
                return _calling_error(procedure, "The active drawable is empty.")

            if not os.path.isfile(MODEL_PATH):
                return _calling_error(
                    procedure,
                    f"NAFNet model is missing: {MODEL_PATH}. Reinstall the plug-in.",
                )

            has_alpha = _drawable_has_alpha(drawable)
            _log(
                f"region-restore: drawable={width}x{height} "
                f"has_alpha={has_alpha} "
                f"image.get_base_type()={image.get_base_type()!r} "
                f"selection={sel_x},{sel_y} {sel_w}x{sel_h}"
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

            _log(
                f"region-restore: ROI {roi_x},{roi_y} {roi_w}x{roi_h} "
                f"(selection + {ctx}px context)"
            )

            _safe_progress(Gimp.progress_init, "NAFNet Restore Region")
            progress_started = True
            _phase("Preparing selection...", 0.05)

            try:
                with tempfile.TemporaryDirectory(prefix="gimp-nafnet-region-") as temp_dir:
                    image_path = os.path.join(temp_dir, "region.png")
                    output_path = os.path.join(temp_dir, "result.png")

                    _phase("Cropping to selection + context...", 0.10)
                    # Crop the drawable's shadow buffer to the ROI
                    # and save. GEGL's buffer-source + gegl:rectangle
                    # handles the sub-region extraction. We also
                    # extract the alpha channel separately (when
                    # the drawable has one) so the worker can
                    # preserve it through the round-trip.
                    full_shadow = drawable.get_shadow_buffer()
                    _log(f"region-restore: full_shadow.get_format()={full_shadow.get_format()!r}")
                    graph = Gegl.Node()
                    source = graph.create_child("gegl:buffer-source")
                    source.set_property("buffer", full_shadow)
                    crop = graph.create_child("gegl:rectangle")
                    crop.set_property("x", float(roi_x))
                    crop.set_property("y", float(roi_y))
                    crop.set_property("width", float(roi_w))
                    crop.set_property("height", float(roi_h))
                    crop.link(source)
                    crop.process()
                    cropped = crop.get_property("buffer")
                    _log(f"region-restore: cropped.get_format()={cropped.get_format()!r}")
                    # Save the cropped region. `save_buffer_as_png`
                    # writes whatever format the buffer is in (RGBA
                    # if the drawable is RGBA, RGB otherwise). The
                    # worker reads RGB channels; the original alpha
                    # is preserved separately via the alpha.png
                    # side channel.
                    save_buffer_as_png(cropped, image_path)

                    image_alpha = None
                    if has_alpha:
                        # RGBA image: extract the alpha channel so
                        # the worker can combine it with the model's
                        # RGB output. For RGB images there's no
                        # alpha component to extract; calling
                        # `gegl:component-extract` on an RGB buffer
                        # fails with "Invalid type".
                        image_alpha = os.path.join(temp_dir, "alpha.png")
                        extract_alpha_png(cropped, image_alpha)
                        _log(f"region-restore: alpha extracted to {image_alpha}")
                    else:
                        _log("region-restore: no alpha (RGB image), skipping alpha round-trip")

                    _phase("Starting NAFNet worker...", 0.20)
                    try:
                        command, use_rust = self._resolve_worker_command(
                            image_path, output_path, alpha_path=image_alpha,
                        )
                    except FileNotFoundError as exc:
                        return _execution_error(procedure, str(exc))
                    except RuntimeError as exc:
                        return _calling_error(procedure, str(exc))

                    _log(f"region-restore: command ({len(command)} args): {' '.join(command)}")
                    if use_rust:
                        _log("worker: rust")
                    else:
                        _log("worker: python")

                    try:
                        completed = _run_worker_with_progress(
                            command, _worker_progress_callback
                        )
                    except OSError as exc:
                        return _execution_error(
                            procedure,
                            f"Could not start the NAFNet worker: {exc}",
                        )

                    # Log the worker's full output (marker lines) for
                    # the diagnostic log. On failure, the stderr is
                    # already in completed.stdout (we merged them).
                    if completed.stdout:
                        for line in completed.stdout.splitlines():
                            if line.strip():
                                _log(f"worker: {line.strip()}")

                    if completed.timed_out:
                        return _execution_error(
                            procedure,
                            f"The NAFNet worker timed out after "
                            f"{WORKER_TIMEOUT_SECONDS} seconds.",
                        )
                    if completed.returncode != 0:
                        return _execution_error(
                            procedure,
                            "The NAFNet worker failed: "
                            + _process_detail(completed),
                        )
                    if not os.path.isfile(output_path):
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
                    shadow = drawable.get_shadow_buffer()
                    try:
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
                    except ValueError as exc:
                        return _execution_error(procedure, str(exc))

                drawable.merge_shadow(True)
                # Update only the original selection bbox — the
                # GIMP display refresh is bounded to what the user
                # actually selected. The result inside the
                # selection is the restored RGB; pixels outside are
                # untouched.
                drawable.update(sel_x, sel_y, sel_w, sel_h)
                Gimp.displays_flush()
            except (OSError, RuntimeError, ValueError) as exc:
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
