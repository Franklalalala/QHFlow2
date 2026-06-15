import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import custom_logger


def test_get_logger_does_not_clear_root_handlers_for_library_imports():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    original_propagate = root.propagate
    original_global_logger = custom_logger._global_logger
    sentinel = logging.NullHandler()

    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.addHandler(sentinel)
    root.setLevel(logging.WARNING)
    root.propagate = True
    custom_logger._global_logger = None

    try:
        logger = custom_logger.get_logger("library_import_smoke")

        assert logger.name.endswith("library_import_smoke")
        assert root.handlers == [sentinel]
        assert root.level == logging.WARNING
        assert root.propagate is True
    finally:
        active_logger = custom_logger._global_logger
        if active_logger is not None:
            for handler in active_logger.handlers[:]:
                active_logger.removeHandler(handler)
                handler.close()
        custom_logger._global_logger = original_global_logger
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)
        root.propagate = original_propagate
