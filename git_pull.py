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

def sync_changes():
    """Sync changes between local and remote repositories.
    
    This handles the scenario where:
    1. Changes were made online (on GitHub)
    2. Changes were also made locally
    3. You want to merge both sets of changes and update the online repository
    """
    print("Starting synchronization between local and remote repositories...")
    
    # Check current branch
    branch_output, branch_success = run_command("git branch --show-current", verbose=False)
    if not branch_success:
        print("Failed to determine current branch.")
        return False
    
    current_branch = branch_output.strip()
    print(f"Current branch: {current_branch}")
    
    # Check if there are local changes that need to be committed
    has_changes = check_for_changes()
    if has_changes:
        print("\n📝 You have uncommitted local changes.")
        try:
            should_commit = input("Would you like to commit these changes first? (y/n): ")
            if should_commit.lower() == 'y':
                commit_msg = input("Enter commit message (or press Enter for default message): ")
                if not commit_msg:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    commit_msg = f"Local changes before sync - {timestamp}"
                
                # Stage and commit changes
                print("\nStaging local changes...")
                _, stage_success = run_command("git add .")
                if not stage_success:
                    print("Failed to stage changes.")
                    return False
                
                print(f"\nCommitting with message: '{commit_msg}'")
                _, commit_success = run_command(f'git commit -m "{commit_msg}"')
                if not commit_success:
                    print("Failed to commit changes.")
                    return False
                
                print("✅ Local changes committed successfully.")
            else:
                print("\n⚠️ Continuing without committing local changes.")
                print("This may make merging more complex.")
        except KeyboardInterrupt:
            print("\nSync operation cancelled.")
            return False
    
    # Fetch remote changes
    print("\nFetching changes from remote repository...")
    _, fetch_success = run_command("git fetch origin")
    if not fetch_success:
        print("Failed to fetch from remote repository.")
        return False
    
    # Check if we need to pull
    behind_check, _ = run_command(f"git rev-list --count HEAD..origin/{current_branch}", verbose=False)
    commits_behind = int(behind_check.strip()) if behind_check.strip().isdigit() else 0
    
    if commits_behind > 0:
        print(f"\nYour local repository is {commits_behind} commit(s) behind the remote.")
        print("Pulling remote changes...")
        
        # Pull changes
        _, pull_success = run_command(f"git pull origin {current_branch}")
        if not pull_success:
            print("\n⚠️ Pull failed. You may need to resolve conflicts manually.")
            print("After resolving conflicts, run:")
            print("  git add .")
            print("  git commit -m \"Resolve merge conflicts\"")
            print("  git push origin " + current_branch)
            return False
        
        print("✅ Successfully pulled remote changes.")
    else:
        print("\nNo new changes to pull from remote repository.")
    
    # Check if we need to push
    ahead_check, _ = run_command(f"git rev-list --count origin/{current_branch}..HEAD", verbose=False)
    commits_ahead = int(ahead_check.strip()) if ahead_check.strip().isdigit() else 0
    
    if commits_ahead > 0:
        print(f"\nYour local repository is {commits_ahead} commit(s) ahead of the remote.")
        print("Pushing local changes to remote...")
        
        # Push changes
        _, push_success = run_command(f"git push origin {current_branch}")
        if not push_success:
            print("Failed to push changes to remote repository.")
            return False
        
        print("✅ Successfully pushed local changes to remote repository.")
    else:
        print("\nNo local commits to push to remote repository.")
    
    print("\n🔄 Synchronization complete! Local and remote repositories are now in sync.")
    return True

def pull_specific_files(file_paths):
    """Pull only specific files from the remote repository.
    
    Args:
        file_paths: List of file paths to pull
    """
    if not file_paths:
        print("No files specified to pull.")
        return False
    
    # Check current branch
    branch_output, branch_success = run_command("git branch --show-current", verbose=False)
    if not branch_success:
        print("Failed to determine current branch.")
        return False
    
    current_branch = branch_output.strip()
    print(f"Current branch: {current_branch}")
    
    # Fetch latest from remote
    print("\nFetching latest changes from remote repository...")
    _, fetch_success = run_command("git fetch origin")
    if not fetch_success:
        print("Failed to fetch from remote repository.")
        return False
    
    # Check if any of the specified files have changes
    modified_files = []
    print("\nChecking for changes in specified files:")
    for file_path in file_paths:
        # Normalize file path
        file_path = file_path.strip()
        print(f"  - {file_path}")
        
        # Check if file exists in remote
        file_check, _ = run_command(f"git ls-tree -r --name-only origin/{current_branch} | grep -q '^{file_path}$' && echo 'exists' || echo 'not found'", verbose=False)
        if "not found" in file_check:
            print(f"    ⚠️ File not found in remote repository: {file_path}")
            continue
        
        # Check if file has changes
        diff_check, _ = run_command(f"git diff --name-only HEAD origin/{current_branch} -- {file_path}", verbose=False)
        if diff_check.strip():
            modified_files.append(file_path)
            print(f"    ✓ Has changes in remote")
        else:
            print(f"    - No changes detected")
    
    if not modified_files:
        print("\n✅ No changes found in the specified files. Everything is up-to-date.")
        return True
    
    # Pull specific files
    print(f"\nPulling {len(modified_files)} modified files from remote repository...")
    
    success = True
    for file_path in modified_files:
        print(f"\nUpdating {file_path}...")
        
        # Check if local file has changes
        local_changes, _ = run_command(f"git diff --name-only -- {file_path}", verbose=False)
        if local_changes.strip():
            print(f"  ⚠️ File has local modifications: {file_path}")
            try:
                choice = input("  What would you like to do? [s]kip, [o]verwrite local changes, [m]erge: ")
                if choice.lower() == 's':
                    print(f"  Skipping {file_path}")
                    continue
                elif choice.lower() == 'o':
                    _, reset_success = run_command(f"git checkout HEAD -- {file_path}")
                    if not reset_success:
                        print(f"  Failed to reset {file_path}")
                        success = False
                        continue
                elif choice.lower() != 'm':
                    print(f"  Invalid choice, skipping {file_path}")
                    continue
            except KeyboardInterrupt:
                print("\nPull operation cancelled.")
                return False
        
        # Checkout the file from remote
        _, checkout_success = run_command(f"git checkout origin/{current_branch} -- {file_path}")
        if not checkout_success:
            print(f"  Failed to checkout {file_path} from remote.")
            success = False
            continue
        
        print(f"  ✅ Updated {file_path} successfully.")
    
    if success:
        print("\n✅ Successfully pulled specified files from the remote repository!")
    else:
        print("\n⚠️ Completed with some errors. Some files may not have been updated.")
    
    return success

if __name__ == "__main__":
    # Parse arguments
    force = False
    hard_reset = False
    sync = False
    specific_files = []
    
    # Process flags first
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ["--force", "-f"]:
            force = True
        elif sys.argv[i] in ["--hard-reset", "-hr"]:
            hard_reset = True
        elif sys.argv[i] in ["--sync", "-s"]:
            sync = True
        elif sys.argv[i] in ["--files", "-files", "--file", "-file"]:
            # Next arguments should be file paths until we hit another flag
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith("-"):
                specific_files.append(sys.argv[i])
                i += 1
            i -= 1  # Compensate for the i++ at the end of the loop
        else:
            # If not a recognized flag, assume it's a file path
            if not sys.argv[i].startswith("-"):
                specific_files.append(sys.argv[i])
        i += 1
    
    # Choose the appropriate operation
    if specific_files:
        # Pull specific files
        pull_specific_files(specific_files)
    elif sync:
        # Sync local and remote changes
        sync_changes()
    else:
        # Regular pull operation
        pull_updates(force=force, hard_reset=hard_reset)
