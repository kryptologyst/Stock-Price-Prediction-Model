#!/usr/bin/env python3
"""Setup script for the stock price prediction project."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        command: Command to run
        description: Description of what the command does
        
    Returns:
        True if command succeeded, False otherwise
    """
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible.
    
    Returns:
        True if Python version is compatible
    """
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✓ Python version {sys.version.split()[0]} is compatible")
    return True


def create_directories() -> None:
    """Create necessary directories."""
    directories = [
        "data",
        "models", 
        "assets",
        "logs",
        "assets/plots",
        "assets/models",
        "assets/results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def install_dependencies() -> bool:
    """Install project dependencies.
    
    Returns:
        True if installation succeeded
    """
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("Error: requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def setup_pre_commit() -> bool:
    """Setup pre-commit hooks.
    
    Returns:
        True if setup succeeded
    """
    # Install pre-commit
    if not run_command("pip install pre-commit", "Installing pre-commit"):
        return False
    
    # Install hooks
    return run_command("pre-commit install", "Installing pre-commit hooks")


def run_tests() -> bool:
    """Run the test suite.
    
    Returns:
        True if tests passed
    """
    return run_command("python -m pytest tests/ -v", "Running tests")


def main():
    """Main setup function."""
    print("Stock Price Prediction Model Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not install_dependencies():
        print("Failed to install dependencies. Please check the error messages above.")
        sys.exit(1)
    
    # Setup pre-commit (optional)
    print("\nSetting up pre-commit hooks...")
    setup_pre_commit()
    
    # Run tests
    print("\nRunning tests...")
    if not run_tests():
        print("Some tests failed. Please check the error messages above.")
        print("You can still proceed, but some functionality may not work correctly.")
    
    print("\n" + "=" * 40)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Run training: python scripts/train.py")
    print("2. Launch demo: streamlit run demo/app.py")
    print("3. Run evaluation: python scripts/evaluate.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
