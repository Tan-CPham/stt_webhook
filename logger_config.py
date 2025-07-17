from loguru import logger
import sys

# Remove default logger configuration
logger.remove()

# Add custom logger configuration (logging to stdout with DEBUG level)
logger.add(sys.stdout, level="DEBUG")

# Optionally, you can log to a file
# logger.add("debug.log", level="DEBUG")
