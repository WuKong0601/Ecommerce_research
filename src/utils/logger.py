"""
Logging utility for CoFARS implementation
"""
import os
import sys
from datetime import datetime
from loguru import logger

def setup_logger(log_dir="logs", log_level="INFO"):
    """
    Setup logger with both file and console outputs
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create log directory if not exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler
    log_file = os.path.join(log_dir, f"cofars_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="100 MB",  # Rotate when file reaches 100MB
        retention="30 days",  # Keep logs for 30 days
        compression="zip"  # Compress rotated logs
    )
    
    logger.info(f"Logger initialized. Logs saved to: {log_file}")
    return logger

# Create global logger instance
def get_logger():
    """Get the global logger instance"""
    return logger
