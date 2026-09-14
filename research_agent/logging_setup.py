from __future__ import annotations

import logging
import sys


LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root = logging.getLogger("research_agent")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
