"""Logging utilities for training and loss landscape analysis."""
import os
import sys
import logging


class ExperimentLogger:
    """Logger for experiment tracking with both file and console output."""

    def __init__(self, output_dir: str, log_filename: str = "experiment.log"):
        """
        Initialize logger.

        Args:
            output_dir: Directory to save log files
            log_filename: Name of the log file
        """
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, log_filename)

        # Create logger
        self.logger = logging.getLogger("ExperimentLogger")
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.logger.handlers = []

        # File handler - captures everything
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler - only important messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

        self.info(f"Log file created: {self.log_file}")

    def debug(self, message: str):
        """Log debug message (file only)."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message (file + console)."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message (file + console)."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message (file + console)."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message (file + console)."""
        self.logger.critical(message)

    def log_section(self, title: str):
        """Log a section separator."""
        separator = "=" * 70
        self.info(f"\n{separator}")
        self.info(title)
        self.info(separator)

    def log_config(self, config_dict: dict, config_name: str = "Configuration"):
        """Log configuration dictionary."""
        self.info(f"\n{config_name}:")
        for key, value in config_dict.items():
            self.info(f"  {key}: {value}")

    def log_metrics(self, metrics: dict, prefix: str = ""):
        """Log metrics dictionary."""
        if prefix:
            self.info(f"\n{prefix}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")

    def log_memory_usage(self):
        """Log current GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                self.debug(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        except ImportError:
            pass

    def flush(self):
        """Flush all handlers."""
        for handler in self.logger.handlers:
            handler.flush()


def setup_logger(output_dir: str, log_filename: str = "experiment.log") -> ExperimentLogger:
    """
    Setup experiment logger.

    Args:
        output_dir: Directory to save log files
        log_filename: Name of the log file

    Returns:
        ExperimentLogger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    return ExperimentLogger(output_dir, log_filename)
