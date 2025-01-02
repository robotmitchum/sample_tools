# coding:utf-8
"""
    :module: logger.py
    :description: Simple logger class
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.12
"""

import logging
import traceback


class SimpleLogger:
    def __init__(self, log_file="error_log.txt"):
        """
        Initialize the logger with an output to a given file
        :param str log_file: Log file path
        """
        self.logger = logging.getLogger("SimpleLogger")
        self.logger.setLevel(logging.ERROR)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)

        # The Formatter method does not support f-string for performance reasons
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def log_exception(self, exception):
        """
        Method to log a full trace
        :param Exception exception:
        """
        error_message = ''.join(
            traceback.format_exception(etype=type(exception), value=exception, tb=exception.__traceback__))
        self.logger.error(f"Exception occurred:\n{error_message}")
