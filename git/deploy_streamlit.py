#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Path to the streamlit app
STREAMLIT_APP_PATH = Path(__file__).parent / "streamlit_app" / "main.py"
REQUIREMENTS_FILE = Path(__file__).parent / "requirements.txt"

def run_command(command, capture_output=False):
    """Run a shell command and optionally return the output."""
    print(f"Running: {command}")
    
    if capture_output:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result
    else:
        # Stream output directly to the console
        return subprocess.run(command, shell=True)

def find_venv():
    """Find a virtual environment in the project directory."""
    project_dir = Path(__file__).parent
    
    # Check common venv locations
    venv_paths = [
        project_dir / "venv",
        project_dir / ".venv",
        project_dir / "env",
        Path(os.environ.get("VIRTUAL_ENV", ""))
    ]
    
    for venv_path in venv_paths:
        if venv_path.exists() and (venv_path / "bin" / "activate").exists():
            return venv_path
        # Check for Windows-style venv
        if venv_path.exists() and (venv_path / "Scripts" / "activate").exists():
            return venv_path
            
    return None

def check_dependencies():
    """Ensure all dependencies are installed."""
    try:
        import streamlit
        print("✓ Streamlit is already installed")
    except ImportError:
        print("Installing Streamlit and dependencies...")
        
        if REQUIREMENTS_FILE.exists():
            run_command(f"pip install -r {REQUIREMENTS_FILE}")
        else:
            run_command("pip install streamlit")

def deploy_streamlit(port=8501, share=False, server_headless=True):
    """Deploy the Streamlit application."""
    if not STREAMLIT_APP_PATH.exists():
        print(f"Error: Streamlit app not found at {STREAMLIT_APP_PATH}")
        print("Make sure the path to your main.py file is correct.")
        return False
    
    # Build the Streamlit command
    command = f"streamlit run {STREAMLIT_APP_PATH}"
    
    # Add arguments
    if port != 8501:
        command += f" --server.port={port}"
    
    if share:
        command += " --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --share"
    elif server_headless:
        command += " --server.headless=true"
        
    # Run Streamlit
    run_command(command)
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Deploy the NHL Playoff Predictor Streamlit app")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on (default: 8501)")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--no-headless", action="store_true", help="Don't run in headless mode")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies, don't deploy")
    args = parser.parse_args()
    
    # Find and activate virtual environment if available
    venv_path = find_venv()
    if venv_path:
        print(f"Found virtual environment at: {venv_path}")
        
        # We can't directly activate a venv from Python
        # Instead, give instructions to the user if not already in a venv
        if not os.environ.get("VIRTUAL_ENV"):
            if os.name == "nt":  # Windows
                activate_path = venv_path / "Scripts" / "activate"
                print(f"\nTo activate the virtual environment, run:")
                print(f"    {activate_path}")
                print("Then run this script again.\n")
            else:  # Unix-like
                activate_path = venv_path / "bin" / "activate"
                print(f"\nTo activate the virtual environment, run:")
                print(f"    source {activate_path}")
                print("Then run this script again.\n")
            
            sys.exit(1)
    else:
        print("No virtual environment found. Using system Python.")
    
    # Check and install dependencies
    check_dependencies()
    
    if args.check_only:
        print("Dependency check complete. Not deploying (--check-only specified).")
        return
    
    # Deploy Streamlit app
    print("\nDeploying NHL Playoff Predictor Streamlit app...")
    deploy_streamlit(
        port=args.port, 
        share=args.share,
        server_headless=not args.no_headless
    )

if __name__ == "__main__":
    main()
