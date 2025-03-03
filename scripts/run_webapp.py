#!/usr/bin/env python
"""
Script to run the Flask web application for EV Opinion Search Engine
"""
import argparse
import os
import sys
import logging

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import app
from config.app_config import (
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run the EV Opinion Search Engine web application')
    parser.add_argument('--host', default=FLASK_HOST, help=f'Host to bind to (default: {FLASK_HOST})')
    parser.add_argument('--port', type=int, default=FLASK_PORT, help=f'Port to bind to (default: {FLASK_PORT})')
    parser.add_argument('--debug', action='store_true', default=FLASK_DEBUG, help='Run in debug mode')

    args = parser.parse_args()

    logger.info(f"Starting web application on {args.host}:{args.port}")
    logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")

    # Run the Flask app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())