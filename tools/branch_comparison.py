#!/usr/bin/env python3
"""
Branch Comparison Tool
Compares differences between main branch and selected branch
"""

import subprocess
import os
import sys
from datetime import datetime
from pathlib import Path
import json

def run_git_command(command, cwd=None):
    """Run git command and return output"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Git command failed: {command}")
            print(f"Error: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error running git command: {e}")
        return None

def get_branch_info(repo_path):
    """Get information about current branches"""
    info = {}
    
    # Get current branch
    current_branch = run_git_command("git branch --show-current", repo_path)
    info['current_branch'] = current_branch
    
    # Get main branch commit
    main_commit = run_git_command("git rev-parse main", repo_path)
    info['main_commit'] = main_commit
    
    # Get selected branch commit
    selected_commit = run_git_command(f"git rev-parse {current_branch}", repo_path)
    info['selected_commit'] = selected_commit
    
    # Get commits ahead/behind
    ahead_behind = run_git_command(f"git rev-list --left-right --count main...{current_branch}", repo_path)
    if ahead_behind:
        behind, ahead = ahead_behind.split('\t')
        info['commits_behind'] = int(behind)
        info['commits_ahead'] = int(ahead)
    
    return info

def get_file_changes(repo_path, base_branch="main"):
    """Get detailed file changes between branches"""
    current_branch = run_git_command("git branch --show-current", repo_path)
    
    # Get list of changed files
    changed_files = run_git_command(
        f"git diff --name-status {base_branch}..{current_branch}", 
        repo_path
    )
    
    changes = {
        'added': [],
        'modified': [],
        'deleted': [],
        'renamed': []
    }
    
    if changed_files:
        for line in changed_files.split('\n'):
            if not line:
                continue
            
            parts = line.split('\t')
            status = parts[0]
            filename = parts[1] if len(parts) > 1 else ''
            
            if status == 'A':
                changes['added'].append(filename)
            elif status == 'M':
                changes['modified'].append(filename)
            elif status == 'D':
                changes['deleted'].append(filename)
            elif status.startswith('R'):
                if len(parts) > 2:
                    changes['renamed'].append(f"{parts[1]} -> {parts[2]}")
                else:
                    changes['renamed'].append(filename)
    
    return changes

def analyze_file_types(changes):
    """Analyze types of files changed"""
    file_types = {}
    
    all_files = (
        changes['added'] + 
        changes['modified'] + 
        changes['deleted']
    )
    
    for file in all_files:
        ext = Path(file).suffix.lower()
        if not ext:
            ext = 'no_extension'
        
        if ext not in file_types:
            file_types[ext] = {'count': 0, 'files': []}
        
        file_types[ext]['count'] += 1
        file_types[ext]['files'].append(file)
    
    return file_types

def get_directory_changes(changes):
    """Analyze changes by directory"""
    dir_changes = {}
    
    all_files = changes['added'] + changes['modified'] + changes['deleted']
    
    for file in all_files:
        dir_path = str(Path(file).parent)
        if dir_path == '.':
            dir_path = 'root'
        
        if dir_path not in dir_changes:
            dir_changes[dir_path] = {'count': 0, 'files': []}
        
        dir_changes[dir_path]['count'] += 1
        dir_changes[dir_path]['files'].append(file)
    
    return dir_changes

def get_code_statistics(repo_path, base_branch="main"):
    """Get code statistics differences"""
    current_branch = run_git_command("git branch --show-current", repo_path)
    
    # Get line changes
    line_stats = run_git_command(
        f"git diff --shortstat {base_branch}..{current_branch}",
        repo_path
    )
    
    stats = {}
    if line_stats:
        # Parse "X files changed, Y insertions(+), Z deletions(-)"
        parts = line_stats.split(', ')
        for part in parts:
            if 'files changed' in part:
                stats['files_changed'] = int(part.split()[0])
            elif 'insertion' in part:
                stats['insertions'] = int(part.split()[0])
            elif 'deletion' in part:
                stats['deletions'] = int(part.split()[0])
    
    return stats

def generate_comparison_report(repo_path, output_file=None):
    """Generate comprehensive comparison report"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"branch_comparison_{timestamp}.md"
    
    print(f"Generating branch comparison report...")
    print(f"Repository: {repo_path}")
    
    # Get branch information
    branch_info = get_branch_info(repo_path)
    print(f"Current branch: {branch_info.get('current_branch', 'unknown')}")
    
    # Get file changes
    file_changes = get_file_changes(repo_path)
    print(f"Files added: {len(file_changes['added'])}")
    print(f"Files modified: {len(file_changes['modified'])}")
    print(f"Files deleted: {len(file_changes['deleted'])}")
    
    # Analyze file types and directories
    file_types = analyze_file_types(file_changes)
    dir_changes = get_directory_changes(file_changes)
    code_stats = get_code_statistics(repo_path)
    
    # Generate report
    report = []
    report.append(f"# Branch Comparison Report")
    report.append(f"")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Repository:** {repo_path}")
    report.append(f"**Base Branch:** main")
    report.append(f"**Selected Branch:** {branch_info.get('current_branch', 'unknown')}")
    report.append(f"")
    
    # Branch information
    report.append(f"## Branch Information")
    report.append(f"")
    report.append(f"- **Main Branch Commit:** {branch_info.get('main_commit', 'unknown')}")
    report.append(f"- **Selected Branch Commit:** {branch_info.get('selected_commit', 'unknown')}")
    report.append(f"- **Commits Behind Main:** {branch_info.get('commits_behind', 0)}")
    report.append(f"- **Commits Ahead of Main:** {branch_info.get('commits_ahead', 0)}")
    report.append(f"")
    
    # Summary statistics
    report.append(f"## Change Summary")
    report.append(f"")
    report.append(f"| Type | Count |")
    report.append(f"|------|-------|")
    report.append(f"| Files Added | {len(file_changes['added'])} |")
    report.append(f"| Files Modified | {len(file_changes['modified'])} |")
    report.append(f"| Files Deleted | {len(file_changes['deleted'])} |")
    report.append(f"| Files Renamed | {len(file_changes['renamed'])} |")
    
    if code_stats:
        report.append(f"| Lines Added | {code_stats.get('insertions', 0)} |")
        report.append(f"| Lines Removed | {code_stats.get('deletions', 0)} |")
    
    report.append(f"")
    
    # File type analysis
    if file_types:
        report.append(f"## Changes by File Type")
        report.append(f"")
        report.append(f"| Extension | Count | Files |")
        report.append(f"|-----------|-------|-------|")
        
        for ext, info in sorted(file_types.items(), key=lambda x: x[1]['count'], reverse=True):
            files_list = ', '.join(info['files'][:3])  # Show first 3 files
            if len(info['files']) > 3:
                files_list += f" (and {len(info['files']) - 3} more)"
            report.append(f"| {ext} | {info['count']} | {files_list} |")
        
        report.append(f"")
    
    # Directory analysis
    if dir_changes:
        report.append(f"## Changes by Directory")
        report.append(f"")
        report.append(f"| Directory | Files Changed |")
        report.append(f"|-----------|---------------|")
        
        for dir_path, info in sorted(dir_changes.items(), key=lambda x: x[1]['count'], reverse=True):
            report.append(f"| {dir_path} | {info['count']} |")
        
        report.append(f"")
    
    # Detailed file lists
    for change_type, files in file_changes.items():
        if files:
            report.append(f"## {change_type.title()} Files")
            report.append(f"")
            for file in sorted(files):
                report.append(f"- {file}")
            report.append(f"")
    
    # Write report
    report_content = '\n'.join(report)
    output_path = os.path.join(repo_path, output_file)
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    print(f"Report generated: {output_path}")
    return output_path

def main():
    """Main function"""
    # Get repository path
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = os.getcwd()
    
    # Get output file
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = None
    
    # Generate report
    try:
        report_path = generate_comparison_report(repo_path, output_file)
        print(f"✅ Branch comparison completed successfully!")
        print(f"📄 Report saved to: {report_path}")
    except Exception as e:
        print(f"❌ Error generating comparison: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()