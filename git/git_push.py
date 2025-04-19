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

def check_if_behind_remote():
    """Check if local branch is behind the remote branch."""
    # Get current branch
    branch_output, branch_success = run_command("git branch --show-current", verbose=False)
    if not branch_success:
        print("Failed to determine current branch.")
        return True, None  # Assume we're behind if we can't check
    
    current_branch = branch_output.strip()
    
    # Fetch from remote to get latest updates
    _, fetch_success = run_command("git fetch origin", verbose=False)
    if not fetch_success:
        print("Failed to fetch from remote repository.")
        return True, None
    
    # Check if we're behind the remote
    behind_check, success = run_command(
        f"git rev-list --count HEAD..origin/{current_branch}", verbose=False
    )
    if not success or not behind_check.strip().isdigit():
        return True, current_branch  # Assume we're behind if check fails
    
    commits_behind = int(behind_check.strip())
    return commits_behind > 0, current_branch

def check_incoming_changes():
    """Check which files have incoming changes from remote."""
    # Get current branch
    branch_output, branch_success = run_command("git branch --show-current", verbose=False)
    if not branch_success:
        return None
    
    current_branch = branch_output.strip()
    
    # Fetch latest changes
    _, fetch_success = run_command("git fetch origin", verbose=False)
    if not fetch_success:
        return None
    
    # Get list of files with incoming changes
    diff_output, diff_success = run_command(
        f"git diff --name-only HEAD..origin/{current_branch}", verbose=False
    )
    if not diff_success:
        return None
    
    return [file for file in diff_output.splitlines() if file.strip()]

def selective_pull(files_to_pull):
    """Pull changes only for specified files."""
    branch_output, branch_success = run_command("git branch --show-current", verbose=False)
    if not branch_success:
        print("Failed to determine current branch.")
        return False
    
    current_branch = branch_output.strip()
    
    success = True
    for file_path in files_to_pull:
        print(f"Pulling changes for {file_path}...")
        _, checkout_success = run_command(f"git checkout origin/{current_branch} -- {file_path}")
        if not checkout_success:
            print(f"  Failed to update {file_path}")
            success = False
        else:
            print(f"  ✅ Updated {file_path}")
    
    return success

def pull_changes(current_branch, strategy="merge"):
    """Pull changes with a specific reconciliation strategy.
    
    Args:
        current_branch: The current branch name
        strategy: One of "merge", "rebase", or "ff-only"
    
    Returns:
        (success, error_message)
    """
    strategy_flag = ""
    if strategy == "merge":
        strategy_flag = "--no-rebase"
    elif strategy == "rebase":
        strategy_flag = "--rebase"
    elif strategy == "ff-only":
        strategy_flag = "--ff-only"
    
    pull_output, pull_success = run_command(f"git pull {strategy_flag} origin {current_branch}")
    
    if not pull_success:
        # Check if this is the divergent branches error
        if "divergent branches" in pull_output or "Need to specify how to reconcile" in pull_output:
            return False, "divergent_branches"
        return False, pull_output
    
    return True, None

def push_updates(commit_message=None, auto_pull=True, force_push=False, selective_files=None, pull_strategy="merge"):
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
    
    # Check if we're behind the remote and need to pull first
    is_behind, current_branch = check_if_behind_remote()
    if is_behind and not force_push:
        print("\n⚠️ Your local branch is behind the remote branch.")
        
        # Check which files have incoming changes
        incoming_files = check_incoming_changes()
        print("\nFiles with incoming changes from remote:")
        if incoming_files:
            for file in incoming_files:
                print(f"  - {file}")
            
            # Specific handling for main.py and requirements.txt
            critical_files = [f for f in incoming_files if f.endswith("main.py") or f == "requirements.txt"]
            if critical_files and not auto_pull:
                print("\nNotice: Critical files have incoming changes:")
                for file in critical_files:
                    print(f"  - {file}")
                
                try:
                    choice = input("\nDo you want to: [1] Update these files, [2] Force push, [3] Cancel? ")
                    if choice == "1":
                        # Pull only these critical files
                        if selective_pull(critical_files):
                            print("\n✅ Successfully updated critical files.")
                            # Re-stage the changes as we've updated files
                            _, _ = run_command("git add .")
                    elif choice == "2":
                        force_push = True
                        print("\n⚠️ Proceeding with force push.")
                    else:
                        print("\nOperation cancelled.")
                        return False
                except KeyboardInterrupt:
                    print("\nOperation cancelled.")
                    return False
        else:
            print("  None found (this is unusual, might be a structural change)")
        
        if auto_pull and not force_push:
            print("\nAttempting to pull changes before pushing...")
            print(f"Using pull strategy: {pull_strategy}")
            
            # Pull changes with specified strategy
            pull_success, error_type = pull_changes(current_branch, strategy=pull_strategy)
            
            if not pull_success:
                print("\n❌ Pull failed.")
                
                # Special handling for divergent branches error
                if error_type == "divergent_branches":
                    print("The issue is that your local and remote branches have diverged.")
                    print("You need to choose how to reconcile the branches:")
                    print("Options:")
                    print("  1. Merge remote changes (recommended if you're unsure):")
                    print("     python git_push.py --strategy merge")
                    print("  2. Rebase your changes on top of remote changes:")
                    print("     python git_push.py --strategy rebase")
                    print("  3. Force push your changes (overwrites remote changes):")
                    print("     python git_push.py --force-push")
                    print("  4. Fetch and merge specific files:")
                    print("     python git_pull.py --files <file1> <file2>")
                    print("     Then: python git_push.py")
                    
                    # Interactively ask what to do
                    try:
                        choice = input("\nChoose an option [1-4]: ")
                        if choice == "1":
                            print("\nTrying merge strategy...")
                            pull_success, _ = pull_changes(current_branch, "merge")
                        elif choice == "2":
                            print("\nTrying rebase strategy...")
                            pull_success, _ = pull_changes(current_branch, "rebase")
                        elif choice == "3":
                            force_push = True
                            print("\n⚠️ Proceeding with force push.")
                            pull_success = True  # Skip pull
                        elif choice == "4" or choice.lower() == "q":
                            print("\nExiting. Run one of the suggested commands to proceed.")
                            return False
                        else:
                            print("\nInvalid choice. Exiting.")
                            return False
                        
                        if not pull_success:
                            print("\n❌ Still unable to reconcile branches. Try resolving conflicts manually.")
                            return False
                    except KeyboardInterrupt:
                        print("\nOperation cancelled.")
                        return False
                else:
                    print("Options:")
                    print("  1. Resolve conflicts manually with: git pull")
                    print("  2. Force push with: python git_push.py --force-push (overwrites remote changes)")
                    return False
            
            print("✅ Successfully pulled remote changes.")
    
    # Push changes
    print("\nPushing to remote repository...")
    push_command = "git push"
    if force_push:
        print("⚠️ Using force push! This will overwrite remote changes.")
        push_command = "git push --force"
    
    _, push_success = run_command(push_command)
    if not push_success:
        print("Failed to push changes to remote repository.")
        print("\nPossible solutions:")
        print("  1. Run git pull to get latest changes, resolve any conflicts, then push")
        print("  2. Run this script with --force-push (use with caution)")
        return False
    
    print("\n✅ Successfully pushed changes to the remote repository!")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    commit_message = None
    auto_pull = True
    force_push = False
    selective_files = None
    pull_strategy = "merge"  # Default strategy
    
    # Process flags
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--no-pull":
            auto_pull = False
        elif sys.argv[i] == "--force-push":
            force_push = True
        elif sys.argv[i] == "--strategy" and i + 1 < len(sys.argv):
            strategy = sys.argv[i+1]
            if strategy in ["merge", "rebase", "ff-only"]:
                pull_strategy = strategy
            else:
                print(f"Invalid strategy: {strategy}. Using 'merge' instead.")
            i += 1
        elif sys.argv[i] == "--files" and i + 1 < len(sys.argv):
            selective_files = sys.argv[i+1:] 
            # Stop processing args as these are all filenames
            break
        else:
            # If not a recognized flag, assume it's part of the commit message
            if commit_message is None:
                commit_message = sys.argv[i]
            else:
                commit_message += " " + sys.argv[i]
        i += 1
    
    push_updates(commit_message, auto_pull, force_push, selective_files, pull_strategy)
