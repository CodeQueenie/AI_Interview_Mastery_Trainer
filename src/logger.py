"""
Centralized Logging Module for AI Interview Mastery Trainer

This module provides a consistent logging interface for all components
of the AI Interview Mastery Trainer application.

Author: Nicole LeGuern (CodeQueenie)
"""

import os
import logging
from datetime import datetime


# Define log levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def get_logger(name, level=logging.INFO):
    """
    Get a logger with the specified name and level.
    
    Args:
        name (str): Name of the logger, typically the module name
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger(f"ai_interview_trainer.{name}")
    
    # Only configure if handlers haven't been added yet
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create log filename with timestamp
        log_filename = f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Set formatter for handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def log_function_call(logger, func_name, args=None, kwargs=None):
    """
    Log a function call with its arguments.
    
    Args:
        logger (logging.Logger): Logger instance
        func_name (str): Name of the function being called
        args (tuple, optional): Positional arguments
        kwargs (dict, optional): Keyword arguments
    """
    args_str = ", ".join([str(arg) for arg in args]) if args else ""
    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
    
    if args_str and kwargs_str:
        params = f"{args_str}, {kwargs_str}"
    elif args_str:
        params = args_str
    elif kwargs_str:
        params = kwargs_str
    else:
        params = ""
    
    logger.debug(f"Calling {func_name}({params})")


def log_exception(logger, e, context=""):
    """
    Log an exception with context.
    
    Args:
        logger (logging.Logger): Logger instance
        e (Exception): The exception to log
        context (str, optional): Additional context information
    """
    if context:
        logger.exception(f"{context}: {str(e)}")
    else:
        logger.exception(str(e))


# Function decorator for logging
def log_function(logger=None, level=logging.DEBUG):
    """
    Decorator to log function calls, arguments, and exceptions.
    
    Args:
        logger (logging.Logger, optional): Logger instance. If None, a logger will be created.
        level (int, optional): Logging level for the function calls
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        # Get function's module and name
        module_name = func.__module__
        func_name = func.__name__
        
        # Create logger if not provided
        nonlocal logger
        if logger is None:
            logger = get_logger(module_name)
        
        def wrapper(*args, **kwargs):
            # Log function call
            if level <= logger.level:
                log_function_call(logger, func_name, args, kwargs)
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Log exception
                log_exception(logger, e, f"Exception in {func_name}")
                raise
        
        return wrapper
    
    return decorator
