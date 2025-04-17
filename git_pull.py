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

def check_for_changes():
    """Check if there are any local changes that might conflict with pull."""
    output, success = run_command("git status --porcelain", verbose=False)
    if not success:
        return True  # Assume there are changes if we can't check
    
    return bool(output.strip())

def pull_updates(force=False, hard_reset=False):
    """Pull updates from the remote git repository.
    
    Args:
        force: If True, pull even if there are local changes
        hard_reset: If True, discard all local changes and reset to remote
    """
    print("Checking current branch...")
    branch_output, branch_success = run_command("git branch --show-current", verbose=False)
    
    if not branch_success:
        print("Failed to determine current branch.")
        return False
    
    current_branch = branch_output.strip()
    print(f"Current branch: {current_branch}")
    
    # If hard reset is requested, display warning and confirmation
    if hard_reset:
        print("\n⚠️ WARNING: Hard reset will DISCARD ALL LOCAL CHANGES and reset to remote branch.")
        print("This action cannot be undone. Uncommitted work will be lost.")
        
        try:
            confirm = input("Type 'reset' to confirm hard reset: ")
            if confirm.lower() != 'reset':
                print("Hard reset cancelled.")
                return False
        except KeyboardInterrupt:
            print("\nHard reset cancelled.")
            return False
            
        # Fetch latest from remote
        print("\nFetching latest changes...")
        _, fetch_success = run_command("git fetch origin")
        if not fetch_success:
            print("Failed to fetch from remote repository.")
            return False
            
        # Perform hard reset
        print(f"\nResetting to origin/{current_branch}...")
        _, reset_success = run_command(f"git reset --hard origin/{current_branch}")
        if not reset_success:
            print("Failed to reset to remote branch.")
            return False
            
        print("\n✅ Successfully reset to remote branch!")
        return True
    
    # Check if there are any uncommitted changes
    has_changes = check_for_changes()
    if has_changes and not force:
        print("\n⚠️ You have uncommitted local changes that might be overwritten.")
        print("Options:")
        print("  1. Commit your changes first with git_push.py")
        print("  2. Stash your changes with: git stash")
        print("  3. Use --force to pull anyway (may cause conflicts)")
        print("  4. Use --hard-reset to discard all local changes and sync with remote")
        return False
    
    # Fetch to see if there are any updates
    print("\nFetching updates from remote repository...")
    _, fetch_success = run_command("git fetch")
    if not fetch_success:
        print("Failed to fetch from remote repository.")
        return False
    
    # Check if we're behind the remote
    behind_check, _ = run_command(f"git rev-list --count HEAD..origin/{current_branch}", verbose=False)
    commits_behind = int(behind_check.strip()) if behind_check.strip().isdigit() else 0
    
    if commits_behind == 0:
        print("\n✅ Already up-to-date. No new changes to pull.")
        return True
    
    # Show what changes will be pulled
    print(f"\nYou are {commits_behind} commit(s) behind the remote repository.")
    
    if commits_behind > 0:
        print("\nIncoming changes:")
        run_command(f"git log --oneline --pretty=format:'  %h - %s (%ar)' HEAD..origin/{current_branch}")
    
    # Pull the changes
    print("\nPulling updates from remote repository...")
    pull_output, pull_success = run_command(f"git pull origin {current_branch}")
    
    if not pull_success:
        print("Failed to pull changes.")
        return False
    
    print("\n✅ Successfully pulled changes from the remote repository!")
    return True

if __name__ == "__main__":
    # Parse arguments
    force = False
    hard_reset = False
    
    if "--force" in sys.argv or "-f" in sys.argv:
        force = True
        # Remove the flag from argv
        sys.argv = [arg for arg in sys.argv if arg != "--force" and arg != "-f"]
    
    if "--hard-reset" in sys.argv or "-hr" in sys.argv:
        hard_reset = True
        # Remove the flag from argv
        sys.argv = [arg for arg in sys.argv if arg != "--hard-reset" and arg != "-hr"]
    
    pull_updates(force=force, hard_reset=hard_reset)
