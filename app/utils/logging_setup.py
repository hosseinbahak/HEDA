# app/utils/logging_setup.py
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

def _json_formatter(record: logging.LogRecord) -> str:
    # Safe JSON formatter
    payload = {
        "level": record.levelname,
        "name": record.name,
        "msg": record.getMessage(),
        "time": getattr(record, "asctime", None) or None,
    }
    if hasattr(record, "extra") and isinstance(record.extra, dict):
        payload.update(record.extra)  # attach structured extras
    return json.dumps(payload, ensure_ascii=False)

class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _json_formatter(record)

def configure_logging(
    name: str = "heda",
    level: Optional[str] = None,
    to_file: Optional[bool] = None,
    log_json: Optional[bool] = None,
    log_path: Optional[str] = None,
):
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    to_file = to_file if to_file is not None else os.getenv("LOG_TO_FILE", "false").lower() == "true"
    log_json = log_json if log_json is not None else os.getenv("LOG_JSON", "false").lower() == "true"
    log_path = log_path or os.getenv("LOG_FILE", "results/heda.log")

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.handlers = []  # reset existing

    if log_json:
        fmt = JsonLogFormatter()
    else:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
        )

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # optional rotating file
    if to_file:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    logger.debug("logging configured", extra={"extra": {
        "level": level, "to_file": to_file, "json": log_json, "file": log_path
    }})
    return logger
