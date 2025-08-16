import logging
import json
import os
from typing import Any, Dict, Optional
from datetime import datetime

class StructuredLogger:
    """A structured logger that outputs both human-readable and JSON formats."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with appropriate handlers."""
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler with human-readable format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # Optional file handler for JSON logs
        log_file = os.getenv('LOG_FILE', 'results/heda.log')
        if log_file:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.error(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.debug(message)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add any extra fields
        if hasattr(record, 'extra_data'):
            log_obj['data'] = record.extra_data
        
        return json.dumps(log_obj)

# Global logger cache
_loggers = {}

def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]

# For backward compatibility with existing code
def configure_logging(
    name: str = "heda",
    level: Optional[str] = None,
    to_file: Optional[bool] = None,
    log_json: Optional[bool] = None,
    log_path: Optional[str] = None,
):
    """Configure logging (backward compatibility function)."""
    logger = get_logger(name)
    
    if level:
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.logger.setLevel(log_level)
    
    return logger.logger