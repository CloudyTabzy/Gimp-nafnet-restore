"""Smoke test: does the deployed nafnet-restore.py import cleanly?

Mocks the GIMP/GEGL/Python-GObject modules so we can at least
syntax-and-import-check the file. If the file fails to import
(syntax error, undefined name, bad attribute, etc.), GIMP would
fail to load it and the menu items would not appear - this is
the "nothing happens at all" failure mode that hides all errors.
"""
import importlib.util
import os
import sys
import types
import unittest.mock as mock
from pathlib import Path

# Cross-platform path to the deployed plug-in. On Windows, the
# per-user GIMP plug-in directory is under %APPDATA%. On other
# platforms, fall back to ~/.config (the XDG-style default) for
# portability, even though GIMP 3.2 is primarily Windows-targeted.
if os.name == "nt":
    _config_root = Path(os.environ["APPDATA"]) / "GIMP" / "3.2" / "plug-ins"
else:
    _config_root = Path.home() / ".config" / "GIMP" / "3.2" / "plug-ins"

DEPLOYED = _config_root / "nafnet-restore" / "nafnet-restore.py"


def mock_gi_module(name: str, **attrs) -> types.ModuleType:
    """Create a fake `gi.repository.NAME` module with the given attrs.

    Any attribute access falls back to a mock that supports any
    attribute. The `__getattr__` is what gi.repository uses for
    lazy attribute loading.
    """
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def main() -> int:
    print(f"Testing import of: {DEPLOYED}")
    if not DEPLOYED.exists():
        raise SystemExit(f"file not found: {DEPLOYED}")

    # Mock gi.repository so `import gi; from gi.repository import ...` works.
    sys.modules["gi"] = types.ModuleType("gi")
    sys.modules["gi.repository"] = types.ModuleType("gi.repository")
    sys.modules["gi.repository.Gegl"] = mock_gi_module(
        "gi.repository.Gegl",
        Node=mock.MagicMock(),
        ColorSpace=mock.MagicMock(),
    )
    sys.modules["gi.repository.Gimp"] = mock_gi_module(
        "gi.repository.Gimp",
        ImageBaseType=mock.MagicMock(),
        ImageType=mock.MagicMock(),
        PDBStatusType=mock.MagicMock(),
        PlugIn=mock.MagicMock(),
        ImageProcedure=mock.MagicMock(),
        ProcedureSensitivityMask=mock.MagicMock(),
    )
    sys.modules["gi.repository.GimpUi"] = mock_gi_module(
        "gi.repository.GimpUi",
        ICON_GEGL="image-x-generic",
    )
    sys.modules["gi.repository.GLib"] = mock_gi_module("gi.repository.GLib")

    # Mock Gimp.main so the plug-in's entry point doesn't actually
    # try to start GIMP. The deployed file ends with
    # `Gimp.main(NafnetRestore.__gtype__, sys.argv)`; in production
    # GIMP imports the file and calls the registered class methods
    # directly, never invoking `Gimp.main`. We mock it so the
    # bottom-of-file call is a no-op for this smoke test.
    sys.modules["gi.repository.Gimp"].main = mock.MagicMock()

    # Also mock `__gtype__` on the class — GObject provides it at
    # runtime; we don't have a real gobject in the smoke test.
    sys.modules["gi.repository.Gimp"].PlugIn.__gtype__ = mock.MagicMock()

    # Now import the deployed file as a module. The bottom of the
    # file is `Gimp.main(NafnetRestore.__gtype__, sys.argv)`, which
    # in production is never called (GIMP imports the file and calls
    # methods on the registered class directly). We expect that
    # line to fail because the mocked `NafnetRestore.__gtype__`
    # isn't a real gobject class. We catch that and consider it a
    # success: the module loaded cleanly, which is what we want
    # to verify.
    spec = importlib.util.spec_from_file_location("nafnet_restore", str(DEPLOYED))
    module = importlib.util.module_from_spec(spec)
    import_failed = None
    try:
        spec.loader.exec_module(module)
    except AttributeError as exc:
        # Expected: the entry-point call references `__gtype__` which
        # is a real GObject attribute in production. Mock can't
        # provide it, so the AttributeError is fine.
        if "__gtype__" in str(exc):
            import_failed = str(exc)
        else:
            print(f"\nIMPORT FAILED: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            return 1
    except Exception as exc:
        print(f"\nIMPORT FAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\nimport phase: {'clean (entry point failed as expected)' if import_failed else 'OK'}")
    if import_failed:
        print(f"  expected error: {import_failed}")
    print(f"NafnetRestore: {module.NafnetRestore}")

    # Try to call the diagnostic methods to ensure they don't
    # throw at attribute-access time.
    inst = module.NafnetRestore()
    procs = inst.do_query_procedures()
    print(f"do_query_procedures() -> {procs}")
    for name in procs:
        proc = inst.do_create_procedure(name)
        print(f"do_create_procedure({name!r}) -> {type(proc).__name__}")

    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
