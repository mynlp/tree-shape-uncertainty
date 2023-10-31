import sys
from logging import DEBUG, INFO, Formatter, Logger, StreamHandler, getLogger
from typing import TextIO

from . import naming


def get_main_logger() -> tuple[Logger, "StreamHandler[TextIO]"]:
    logger = getLogger(naming.main_logger_name())
    logger.setLevel(DEBUG)
    handler = StreamHandler()
    handler.setStream(sys.stderr)
    # handler.setLevel(DEBUG)
    handler.setLevel(INFO)
    fmt = Formatter("%(asctime)s %(name)s: [%(levelname)s]: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False

    return logger, handler


def set_debug_mode(handler: "StreamHandler[TextIO]", debug: bool) -> None:
    if debug == True:
        handler.setLevel(DEBUG)
        fmt = Formatter(
            "%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"
        )
        handler.setFormatter(fmt)


main_logger, main_handler = get_main_logger()
