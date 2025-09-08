#!/usr/bin/env python3
"""
Quick Branch Comparison Utility
Provides a simple comparison view between branches
"""

import subprocess
import os
import sys
from collections import defaultdict

def run_cmd(cmd, cwd=None):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        return result.stdout.strip() if result.returncode == 0 else None
    except:
        return None

def quick_comparison(repo_path="."):
    """Generate quick comparison summary"""
    
    current_branch = run_cmd("git branch --show-current", repo_path)
    if not current_branch:
        print("❌ Not in a git repository")
        return
    
    print(f"🔍 Quick Branch Comparison")
    print(f"{'='*50}")
    print(f"📍 Current Branch: {current_branch}")
    print(f"📍 Comparing with: main")
    print()
    
    # Get basic stats
    stats = run_cmd(f"git diff --shortstat main..{current_branch}", repo_path)
    if stats:
        print(f"📊 Change Statistics:")
        print(f"   {stats}")
        print()
    
    # Get file counts
    added = run_cmd(f"git diff --name-only --diff-filter=A main..{current_branch}", repo_path)
    modified = run_cmd(f"git diff --name-only --diff-filter=M main..{current_branch}", repo_path)
    deleted = run_cmd(f"git diff --name-only --diff-filter=D main..{current_branch}", repo_path)
    
    added_count = len(added.split('\n')) if added else 0
    modified_count = len(modified.split('\n')) if modified else 0
    deleted_count = len(deleted.split('\n')) if deleted else 0
    
    print(f"📁 File Changes:")
    print(f"   ✅ Added:    {added_count} files")
    print(f"   📝 Modified: {modified_count} files") 
    print(f"   ❌ Deleted:  {deleted_count} files")
    print()
    
    # Top directories with changes
    all_files = []
    if added:
        all_files.extend(added.split('\n'))
    if modified:
        all_files.extend(modified.split('\n'))
    if deleted:
        all_files.extend(deleted.split('\n'))
    
    if all_files:
        dir_counts = defaultdict(int)
        for file in all_files:
            if file:  # Skip empty lines
                dir_name = os.path.dirname(file) or 'root'
                dir_counts[dir_name] += 1
        
        print(f"📂 Top Changed Directories:")
        for dir_name, count in sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {dir_name}: {count} files")
        print()
    
    # Show some example new files
    if added:
        new_files = added.split('\n')[:10]  # First 10
        print(f"📄 Sample New Files:")
        for file in new_files:
            if file:
                print(f"   + {file}")
        if added_count > 10:
            print(f"   ... and {added_count - 10} more")
        print()
    
    # Show commits ahead/behind
    ahead_behind = run_cmd(f"git rev-list --left-right --count main...{current_branch}", repo_path)
    if ahead_behind:
        behind, ahead = ahead_behind.split('\t')
        print(f"🚀 Branch Status:")
        print(f"   📈 Commits ahead of main: {ahead}")
        print(f"   📉 Commits behind main: {behind}")
        print()
    
    print(f"✨ Summary:")
    if added_count > 50:
        print(f"   🔥 Major restructuring detected ({added_count} new files)")
    elif added_count > 10:
        print(f"   📈 Significant changes ({added_count} new files)")
    else:
        print(f"   📝 Minor changes ({added_count} new files)")
    
    if added_count > modified_count * 2:
        print(f"   🏗️  Appears to be a refactoring/restructuring effort")
    elif modified_count > added_count:
        print(f"   🔧 Appears to be primarily modifications to existing code")
    
    print()
    print(f"💡 Tip: Use 'python tools/branch_comparison.py' for detailed analysis")

if __name__ == "__main__":
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    quick_comparison(repo_path)