"""
Script to run all analysis tools and generate a comprehensive report
for cross-file consistency review.
"""

import os
import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime
import importlib

def run_analysis():
    """Run all analysis tools and generate a consolidated report."""
    # Get the project root directory - path to the tools directory
    tools_dir = str(Path(__file__).parent)
    
    # Create analysis output directory if it doesn't exist
    analysis_dir = os.path.join(tools_dir, 'analysis_output')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Timestamp for this analysis run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(analysis_dir, timestamp)
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"Running comprehensive analysis...")
    
    # Run import analysis
    try:
        print("\n=== Running import analysis ===")
        # First check if module exists
        import_analysis_path = os.path.join(tools_dir, 'analyze_imports.py')
        if not os.path.exists(import_analysis_path):
            print(f"Import analysis tool not found at {import_analysis_path}")
        else:
            # Pass the project root directory (one level up from tools)
            project_root = str(Path(tools_dir).parent)
            # Pass the analysis output directory as a parameter
            subprocess.run([sys.executable, import_analysis_path, project_root, report_dir], check=True)
    except Exception as e:
        print(f"Error running import analysis: {e}")
    
    # Run function analysis
    try:
        print("\n=== Running function analysis ===")
        function_analysis_path = os.path.join(tools_dir, 'analyze_functions.py')
        if not os.path.exists(function_analysis_path):
            print(f"Function analysis tool not found at {function_analysis_path}")
        else:
            # Pass the project root directory and the report directory
            project_root = str(Path(tools_dir).parent)
            subprocess.run([sys.executable, function_analysis_path, project_root, report_dir], check=True)
    except Exception as e:
        print(f"Error running function analysis: {e}")
    
    # Run constants validator
    try:
        print("\n=== Running constants validator ===")
        constants_validator_path = os.path.join(tools_dir, 'constants_validator.py')
        if not os.path.exists(constants_validator_path):
            print(f"Constants validator not found at {constants_validator_path}")
        else:
            # Pass the project root directory and the report directory
            project_root = str(Path(tools_dir).parent)
            subprocess.run([sys.executable, constants_validator_path, project_root, report_dir], check=True)
    except Exception as e:
        print(f"Error running constants validator: {e}")
    
    # Run dependency analyzer
    try:
        print("\n=== Running dependency analyzer ===")
        dependency_analyzer_path = os.path.join(tools_dir, 'dependency_analyzer.py')
        if not os.path.exists(dependency_analyzer_path):
            print(f"Dependency analyzer not found at {dependency_analyzer_path}")
        else:
            # Pass the project root directory and the report directory
            project_root = str(Path(tools_dir).parent)
            subprocess.run([sys.executable, dependency_analyzer_path, project_root, report_dir], check=True)
    except Exception as e:
        print(f"Error running dependency analyzer: {e}")
    
    # Generate summary report
    try:
        print("\n=== Generating summary report ===")
        
        # Check if we have the needed files
        constants_summary_path = os.path.join(report_dir, 'constants_report.md')
        functions_summary_path = os.path.join(report_dir, 'function_report.md')
        imports_summary_path = os.path.join(report_dir, 'constants_analysis.html')
        
        # Generate the summary report
        with open(os.path.join(report_dir, 'summary_report.md'), 'w') as f:
            f.write(f"# NHL Playoff Model - Codebase Analysis Summary\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            f.write("## Overview\n\n")
            f.write("This report consolidates findings from multiple analysis tools:\n\n")
            f.write("1. **Import Analysis** - Examines how modules and constants are imported across files\n")
            f.write("2. **Function Analysis** - Identifies duplicated and inconsistent function implementations\n") 
            f.write("3. **Constants Validator** - Checks for consistent usage of critical constants\n")
            f.write("4. **Dependency Analyzer** - Maps relationships between files and modules\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Include snippets from individual reports if they exist
            if os.path.exists(constants_summary_path):
                f.write("### Constants Issues\n\n")
                with open(constants_summary_path, 'r') as constants_file:
                    content = constants_file.read()
                    # Extract critical issues section
                    if "Critical Constants Defined Outside config.py" in content:
                        start = content.find("### Critical Constants Defined Outside config.py")
                        end = content.find("###", start + 10)
                        if end == -1:
                            end = len(content)
                        f.write(content[start:end] + "\n\n")
                    else:
                        f.write("No critical constants issues found.\n\n")
            
            if os.path.exists(functions_summary_path):
                f.write("### Function Duplication Issues\n\n")
                with open(functions_summary_path, 'r') as functions_file:
                    content = functions_file.read()
                    # Extract duplicated functions section
                    if "## Functions Defined in Multiple Files" in content:
                        start = content.find("## Functions Defined in Multiple Files")
                        end = content.find("##", start + 10)
                        if end == -1:
                            end = len(content)
                        f.write(content[start:end] + "\n\n")
                    else:
                        f.write("No function duplication issues found.\n\n")
            
            # Add recommendations section
            f.write("## Recommendations\n\n")
            f.write("Based on the analysis, the following actions are recommended:\n\n")
            f.write("1. **Centralize Critical Constants** - Move all critical constants to config.py\n")
            f.write("2. **Eliminate Duplicate Functions** - Consolidate duplicate implementations\n")
            f.write("3. **Standardize Import Patterns** - Ensure consistent import patterns\n")
            f.write("4. **Refactor Complex Functions** - Break down complex functions highlighted in the analysis\n\n")
            
            f.write("## Detailed Reports\n\n")
            f.write("For more detailed analysis, see the following reports in the same directory:\n\n")
            
            if os.path.exists(constants_summary_path):
                f.write("- [Constants Analysis](constants_report.md)\n")
            
            if os.path.exists(functions_summary_path):
                f.write("- [Function Analysis](function_report.md)\n")
            
            if os.path.exists(imports_summary_path):
                f.write("- [Imports Analysis](constants_analysis.html)\n")
            
            if os.path.exists(os.path.join(report_dir, 'analysis_report.md')):
                f.write("- [Dependency Analysis](analysis_report.md)\n")
    except Exception as e:
        print(f"Error generating summary report: {e}")
    
    print(f"\nAnalysis complete. Reports available in {report_dir}")
    return report_dir

if __name__ == "__main__":
    report_dir = run_analysis()
    print(f"Analysis results saved to {report_dir}")
