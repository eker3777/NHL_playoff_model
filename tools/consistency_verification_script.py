"""
Script to verify consistent usage of constants across files.
Run this after making changes to ensure all files are using constants correctly.
"""

import os
import re
import sys
from pathlib import Path

def check_file_for_constants(file_path):
    """Check if a file properly imports and uses constants."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for proper imports
    has_config_import = re.search(r'from\s+streamlit_app\.config\s+import', content) is not None
    
    # Check for HOME_ICE_ADVANTAGE
    has_home_ice_import = re.search(r'from\s+streamlit_app\.config\s+import\s+[^)]*HOME_ICE_ADVANTAGE', content) is not None
    
    # More specific pattern for hardcoded home ice values
    hardcoded_home_ice_pattern = r'(?:home_ice\w*|ice_advantage|ice_boost|ice_factor)\s*=\s*0\.0(3|4)\d*'
    hardcoded_home_ice = re.search(hardcoded_home_ice_pattern, content) is not None
    
    if hardcoded_home_ice:
        home_ice_matches = re.finditer(hardcoded_home_ice_pattern, content)
        hardcoded_home_ice_values = [match.group(0) for match in home_ice_matches]
    else:
        hardcoded_home_ice_values = []
    
    # Check for SERIES_LENGTH_DISTRIBUTION
    has_series_dist_import = re.search(r'from\s+streamlit_app\.config\s+import\s+[^)]*SERIES_LENGTH_DISTRIBUTION', content) is not None
    
    # More specific pattern for hardcoded series distributions
    hardcoded_series_dist_pattern = r'\[\s*0\.\d+\s*,\s*0\.\d+\s*,\s*0\.\d+\s*,\s*0\.\d+\s*\]'
    hardcoded_series_dist = re.search(hardcoded_series_dist_pattern, content) is not None
    
    if hardcoded_series_dist:
        series_dist_matches = re.finditer(hardcoded_series_dist_pattern, content)
        series_dist_values = [match.group(0) for match in series_dist_matches]
    else:
        series_dist_values = []
    
    return {
        'file': file_path,
        'has_config_import': has_config_import,
        'has_home_ice_import': has_home_ice_import,
        'hardcoded_home_ice': hardcoded_home_ice,
        'hardcoded_home_ice_values': hardcoded_home_ice_values,
        'has_series_dist_import': has_series_dist_import,
        'hardcoded_series_dist': hardcoded_series_dist,
        'hardcoded_series_dist_values': series_dist_values
    }

def main():
    """Main function to check files."""
    project_root = Path('/workspaces/NHL_playoff_model')
    
    # Files to check
    files_to_check = [
        project_root / 'streamlit_app' / 'models' / 'simulation.py',
        project_root / 'streamlit_app' / 'pages' / 'first_round.py',
        project_root / 'streamlit_app' / 'pages' / 'head_to_head.py',
        project_root / 'streamlit_app' / 'pages' / 'sim_bracket.py',
        project_root / 'streamlit_app' / 'utils' / 'debug_utils.py',
        project_root / 'streamlit_app' / 'utils' / 'simulation_utils.py',
    ]
    
    results = []
    for file_path in files_to_check:
        if file_path.exists():
            result = check_file_for_constants(file_path)
            results.append(result)
        else:
            print(f"File not found: {file_path}")
    
    # Display results
    print("\nCONSTANT USAGE VERIFICATION RESULTS\n")
    print("| File | Config Import | HOME_ICE Import | Hardcoded HOME_ICE | SERIES_DIST Import | Hardcoded SERIES_DIST |")
    print("|------|--------------|-----------------|-------------------|-------------------|---------------------|")
    
    for result in results:
        file_name = os.path.basename(result['file'])
        print(f"| {file_name} | {'✅' if result['has_config_import'] else '❌'} | {'✅' if result['has_home_ice_import'] else '❌'} | {'❌' if result['hardcoded_home_ice'] else '✅'} | {'✅' if result['has_series_dist_import'] else '❌'} | {'❌' if result['hardcoded_series_dist'] else '✅'} |")
    
    # Check for any remaining issues
    has_issues = any(
        not r['has_config_import'] or 
        not r['has_home_ice_import'] or 
        r['hardcoded_home_ice'] or 
        not r['has_series_dist_import'] or 
        r['hardcoded_series_dist'] 
        for r in results
    )
    
    if has_issues:
        print("\n⚠️ Issues detected! Some files still have inconsistent constant usage.")
        return 1
    else:
        print("\n✅ All files are using constants consistently.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
