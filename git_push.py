#!/usr/bin/env python3

import os
import sys
import subprocess
from datetime import datetime

def run_command(command, verbose=True):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            shell=True
        )
        if verbose and result.stdout:
            print(result.stdout)
        return result.stdout, True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        return e.stderr, False

def git_status():
    """Check git status and return modified files."""
    output, success = run_command("git status --porcelain", verbose=False)
    if not success:
        return []
    
    modified_files = []
    for line in output.splitlines():
        if line:
            status, filename = line[:2], line[3:]
            modified_files.append((status, filename))
    
    return modified_files

def push_updates(commit_message=None):
    """Push updates to the git repository."""
    print("Checking Git status...")
    modified_files = git_status()
    
    if not modified_files:
        print("No changes to commit.")
        return False
    
    # Print modified files
    print(f"Found {len(modified_files)} modified files:")
    for status, filename in modified_files:
        status_desc = {
            'M ': 'Modified:',
            'A ': 'Added:',
            'D ': 'Deleted:',
            'R ': 'Renamed:',
            'C ': 'Copied:',
            '??': 'Untracked:'
        }.get(status, 'Changed:')
        print(f"  {status_desc} {filename}")
    
    # Use default commit message if none provided
    if not commit_message:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Update NHL playoff model - {timestamp}"
    
    # Stage all changes
    print("\nStaging changes...")
    _, stage_success = run_command("git add .")
    if not stage_success:
        print("Failed to stage changes.")
        return False
    
    # Commit changes
    print("\nCommitting changes...")
    _, commit_success = run_command(f'git commit -m "{commit_message}"')
    if not commit_success:
        print("Failed to commit changes.")
        return False
    
    # Push changes
    print("\nPushing to remote repository...")
    _, push_success = run_command("git push")
    if not push_success:
        print("Failed to push changes to remote repository.")
        return False
    
    print("\n✅ Successfully pushed changes to the remote repository!")
    return True

if __name__ == "__main__":
    # Get optional commit message from command line arguments
    commit_message = None
    if len(sys.argv) > 1:
        commit_message = " ".join(sys.argv[1:])
    
    push_updates(commit_message)
