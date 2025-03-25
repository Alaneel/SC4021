#!/usr/bin/env python3
"""
Setup script for Streaming Opinion Search Engine.
This script sets up the required directories, dependencies, and configuration.
"""

import os
import sys
import subprocess
import argparse
import shutil
import getpass
import json
from pathlib import Path


# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message):
    """Print a formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.ENDC}\n")


def print_success(message):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")


def print_error(message):
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")


def print_info(message):
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")


def run_command(command, description=None, exit_on_error=True, cwd=None):
    """Run a shell command and handle errors."""
    if description:
        print(f"{description}...")

    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0:
        print_error(f"Command failed: {command}")
        print(f"Error: {result.stderr}")
        if exit_on_error:
            sys.exit(1)
        return False
    return True


def check_python_version():
    """Check that Python version is 3.8+."""
    print_header("Checking Python Version")

    if sys.version_info < (3, 8):
        print_error("Python 3.8 or higher is required")
        print(f"Found Python {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)

    print_success(f"Using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def create_directories():
    """Create required directories if they don't exist."""
    print_header("Creating Directories")

    directories = [
        'data/raw',
        'data/processed',
        'classification/models',
        'evaluation/results',
        'logs',
        'tmp'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_success(f"Created {directory}/")


def setup_virtual_environment(force_recreate=False):
    """Set up a virtual environment."""
    print_header("Setting Up Virtual Environment")

    venv_path = 'venv'

    # Check if venv already exists
    if os.path.exists(venv_path) and not force_recreate:
        print_info(f"Virtual environment already exists at {venv_path}")
        return venv_path

    # Remove existing venv if force_recreate is True
    if os.path.exists(venv_path) and force_recreate:
        shutil.rmtree(venv_path)
        print_info("Removed existing virtual environment")

    # Create a new virtual environment
    run_command(f"{sys.executable} -m venv {venv_path}", "Creating virtual environment")
    print_success(f"Created virtual environment at {venv_path}")

    return venv_path


def install_dependencies(venv_path):
    """Install Python dependencies within the virtual environment."""
    print_header("Installing Dependencies")

    # Determine the pip executable based on the platform
    if sys.platform == 'win32':
        pip_exec = f"{venv_path}\\Scripts\\pip"
    else:
        pip_exec = f"{venv_path}/bin/pip"

    # Upgrade pip
    run_command(f"{pip_exec} install --upgrade pip", "Upgrading pip")

    # Install dependencies from requirements.txt
    run_command(f"{pip_exec} install -r requirements.txt", "Installing required packages")
    print_success("Installed Python dependencies")

    # Install NLTK data
    nltk_command = f"{pip_exec} install nltk && " + \
                   f"{sys.executable} -m nltk.downloader punkt stopwords wordnet vader_lexicon " + \
                   "averaged_perceptron_tagger maxent_ne_chunker words"
    run_command(nltk_command, "Downloading NLTK data", exit_on_error=False)

    # Install spaCy model
    spacy_command = f"{pip_exec} install spacy && " + \
                    f"{sys.executable} -m spacy download en_core_web_sm"
    run_command(spacy_command, "Downloading spaCy model", exit_on_error=False)


def setup_crawler_credentials():
    """Set up API credentials for crawlers."""
    print_header("Setting Up Crawler Credentials")

    # Check if credentials.py already exists
    if os.path.exists('crawler/credentials.py'):
        overwrite = input("crawler/credentials.py already exists. Overwrite? (y/n): ").lower() == 'y'
        if not overwrite:
            print_info("Skipping credentials setup")
            return

# Template for credentials
template = """
# Reddit API credentials
# Create an app at https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID = "{reddit_client_id}"
REDDIT_CLIENT_SECRET = "{reddit_client_secret}"
REDDIT_USER_AGENT = "python:streaming_opinions:v1.0 (by /u/{reddit_username})"

# Twitter API credentials
# Create an app at https://developer.twitter.com/en/portal/dashboard
TWITTER_CONSUMER_KEY = "{twitter_consumer_key}"
TWITTER_CONSUMER_SECRET = "{twitter_consumer_secret}"
TWITTER_ACCESS_TOKEN = "{twitter_access_token}"
TWITTER_ACCESS_TOKEN_SECRET = "{twitter_access_token_secret}"
TWITTER_BEARER_TOKEN = "{twitter_bearer_token}"
"""

# Prompt for credentials or use dummy values
use_dummy = input("Would you like to use dummy values for API credentials? (y/n): ").lower() == 'y'

if use_dummy:
    print_info("Using dummy values for API credentials")
    reddit_client_id = "your_reddit_client_id"
    reddit_client_secret = "your_reddit_client_secret"
    reddit_username = "your_username"
    twitter_consumer_key = "your_twitter_consumer_key"
    twitter_consumer_secret = "your_twitter_consumer_secret"
    twitter_access_token = "your_twitter_access_token"
    twitter_access_token_secret = "your_twitter_access_token_secret"
    twitter_bearer_token = "your_twitter_bearer_token"
else:
    print_info("Please enter your API credentials")

    print_info("\nReddit Credentials:")
    reddit_client_id = input("  Client ID: ").strip()
    reddit_client_secret = input("  Client Secret: ").strip()
    reddit_username = input("  Reddit Username: ").strip()

    print_info("\nTwitter Credentials:")
    twitter_consumer_key = input("  Consumer Key: ").strip()
    twitter_consumer_secret = input("  Consumer Secret: ").strip()
    twitter_access_token = input("  Access Token: ").strip()
    twitter_access_token_secret = input("  Access Token Secret: ").strip()
    twitter_bearer_token = input("  Bearer Token: ").strip()

# Fill in the template
credentials_content = template.format(
    reddit_client_id=reddit_client_id,
    reddit_client_secret=reddit_client_secret,
    reddit_username=reddit_username,
    twitter_consumer_key=twitter_consumer_key,
    twitter_consumer_secret=twitter_consumer_secret,
    twitter_access_token=twitter_access_token,
    twitter_access_token_secret=twitter_access_token_secret,
    twitter_bearer_token=twitter_bearer_token
)

# Write to file
with open('crawler/credentials.py', 'w') as f:
    f.write(credentials_content)

print_success("Created crawler/credentials.py")

# Add to .gitignore if not already there
if os.path.exists('.gitignore'):
    with open('.gitignore', 'r') as f:
        gitignore_content = f.read()

    if 'crawler/credentials.py' not in gitignore_content:
        with open('.gitignore', 'a') as f:
            f.write('\n# API credentials\ncrawler/credentials.py\n')
        print_success("Added crawler/credentials.py to .gitignore")


def check_solr_installation():
    """Check if Solr is installed and running."""
    print_header("Checking Solr Installation")

    # Check if Solr is installed
    solr_installed = False

    # Try different methods to detect Solr
    if shutil.which('solr'):
        solr_installed = True
    elif os.path.exists('/opt/solr') or os.path.exists('/usr/local/solr'):
        solr_installed = True
    elif 'SOLR_HOME' in os.environ:
        solr_installed = True

    if not solr_installed:
        print_warning("Solr does not appear to be installed")
        print_info("Please install Apache Solr 9.x from https://solr.apache.org/downloads.html")
        print_info("After installation, create a core called 'streaming_opinions'")
        return False

    # Check if Solr is running
    solr_running = False
    try:
        import requests
        response = requests.get('http://localhost:8983/solr/', timeout=2)
        solr_running = response.status_code == 200
    except:
        solr_running = False

    if not solr_running:
        print_warning("Solr is installed but does not appear to be running")
        print_info("Start Solr with: bin/solr start (Unix) or bin\\solr.cmd start (Windows)")
        return False

    # Check if the core exists
    core_exists = False
    try:
        import requests
        response = requests.get('http://localhost:8983/solr/admin/cores?action=STATUS&core=streaming_opinions',
                                timeout=2)
        data = response.json()
        core_exists = 'streaming_opinions' in data.get('status', {})
    except:
        core_exists = False

    if not core_exists:
        print_warning("Solr core 'streaming_opinions' does not exist")
        print_info("Create the core with: bin/solr create -c streaming_opinions")
        print_info("Then copy the schema.xml file from solr_files/ to the conf directory of your Solr core")
        return False

    print_success("Solr is installed, running, and the 'streaming_opinions' core exists")
    return True


def setup_sample_data():
    """Copy sample data for testing."""
    print_header("Setting Up Sample Data")

    # Check if sample data directory exists
    sample_data_dir = 'sample_data'
    if not os.path.exists(sample_data_dir):
        print_warning("Sample data directory not found")
        return

    # Copy sample data files to data directory
    sample_files = [f for f in os.listdir(sample_data_dir) if f.endswith('.csv')]

    if not sample_files:
        print_warning("No sample CSV files found")
        return

    for file in sample_files:
        src = os.path.join(sample_data_dir, file)
        dst = os.path.join('data', file)
        shutil.copy2(src, dst)
        print_success(f"Copied {src} to {dst}")


def run_tests():
    """Run integration tests to verify setup."""
    print_header("Running Integration Tests")

    # Run tests with unittest
    test_command = f"{sys.executable} -m unittest discover -s tests"
    run_command(test_command, "Running integration tests", exit_on_error=False)


def generate_activation_script(venv_path):
    """Generate an activation script to start the application."""
    print_header("Generating Activation Script")

    # Create activation script for Unix
    if sys.platform != 'win32':
        activate_content = f"""#!/bin/bash
# Activate the virtual environment and start the application
source {venv_path}/bin/activate
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
"""
        with open('start.sh', 'w') as f:
            f.write(activate_content)
        os.chmod('start.sh', 0o755)
        print_success("Created start.sh")

    # Create activation script for Windows
    else:
        activate_content = f"""@echo off
REM Activate the virtual environment and start the application
call {venv_path}\\Scripts\\activate.bat
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
"""
        with open('start.bat', 'w') as f:
            f.write(activate_content)
        print_success("Created start.bat")


def display_next_steps(solr_running):
    """Display the next steps for the user."""
    print_header("Setup Complete")

    print_info("Next Steps:")

    if not solr_running:
        print_info("1. Install and start Apache Solr")
        print_info("2. Create a core called 'streaming_opinions'")
        print_info("3. Copy schema.xml to the Solr core configuration")

    if sys.platform != 'win32':
        print_info("Run './start.sh' to activate the environment and start the application")
    else:
        print_info("Run 'start.bat' to activate the environment and start the application")

    print_info("Access the web interface at http://localhost:5000")

    print_info("\nAdditional Commands:")
    print_info("- Run crawlers:              python -m crawler.run_crawler")
    print_info("- Process data:              python -m processing.processing")
    print_info("- Import to Solr:            python -m indexer.import_to_solr")
    print_info("- Evaluate classifier:       python -m evaluation.evaluate_classifier")
    print_info("- Run performance tests:     python -m evaluation.performance_metrics")
    print_info("- Perform ablation study:    python -m evaluation.ablation_study")


def main():
    """Main function for the setup script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Setup script for Streaming Opinion Search Engine')
    parser.add_argument('--force', action='store_true', help='Force recreation of virtual environment')
    parser.add_argument('--skip-solr-check', action='store_true', help='Skip Solr installation check')
    parser.add_argument('--skip-credentials', action='store_true', help='Skip API credentials setup')
    args = parser.parse_args()

    print_header("Setting Up Streaming Opinion Search Engine")

    # Check Python version
    check_python_version()

    # Create required directories
    create_directories()

    # Set up virtual environment
    venv_path = setup_virtual_environment(args.force)

    # Install dependencies
    install_dependencies(venv_path)

    # Set up API credentials
    if not args.skip_credentials:
        setup_crawler_credentials()

    # Check Solr installation
    solr_running = True
    if not args.skip_solr_check:
        solr_running = check_solr_installation()

    # Set up sample data
    setup_sample_data()

    # Run tests
    run_tests()

    # Generate activation script
    generate_activation_script(venv_path)

    # Display next steps
    display_next_steps(solr_running)


if __name__ == "__main__":
    main()