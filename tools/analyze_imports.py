"""
Utility script to analyze imports and constants usage across the codebase.
This helps identify consistency issues in the NHL Playoff Model.
"""

import os
import re
import sys
from pathlib import Path
import ast
import importlib
import json
from datetime import datetime

def find_python_files(root_dir):
    """Find all Python files in the project."""
    python_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                # Skip this analysis script itself
                if not full_path.endswith('analyze_imports.py'):
                    python_files.append(full_path)
    return python_files

def extract_imports(file_path):
    """Extract all import statements from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    imports = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        'type': 'import',
                        'module': name.name,
                        'alias': name.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    imports.append({
                        'type': 'from',
                        'module': module,
                        'name': name.name,
                        'alias': name.asname
                    })
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return []
        
    return imports

def extract_constants(file_path):
    """Extract constant variable definitions from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    constants = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            # Look for assignments at the module level (not in functions or classes)
            if isinstance(node, ast.Assign) and node.col_offset == 0:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if name is ALL_CAPS (typical for constants)
                        name = target.id
                        if name.isupper() or (len(name) > 3 and '_' in name and name.upper() == name):
                            # Try to get the value
                            try:
                                if isinstance(node.value, ast.Num):
                                    value = node.value.n
                                elif isinstance(node.value, ast.Str):
                                    value = node.value.s
                                elif isinstance(node.value, ast.List):
                                    value = "List[...]"
                                elif isinstance(node.value, ast.Dict):
                                    value = "Dict{...}"
                                elif isinstance(node.value, ast.Tuple):
                                    value = "Tuple(...)"
                                else:
                                    value = str(ast.dump(node.value))
                                
                                constants.append({
                                    'name': name,
                                    'value': value
                                })
                            except:
                                constants.append({
                                    'name': name,
                                    'value': 'unknown'
                                })
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return []
        
    return constants

def generate_report(outdir):
    """Generate a comprehensive report of imports and constants usage across files."""
    # Get the project root directory
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = str(Path(__file__).parent.parent)
    
    # Get the output directory (either provided or default)
    if len(sys.argv) > 2:
        outdir = sys.argv[2]
    
    print(f"Analyzing Python files in {root_dir}")
    
    # Find all Python files
    files = find_python_files(root_dir)
    print(f"Found {len(files)} Python files")
    
    # Analyze each file
    file_analysis = {}
    all_imports = {}
    all_constants = {}
    
    for file_path in files:
        rel_path = os.path.relpath(file_path, root_dir)
        imports = extract_imports(file_path)
        constants = extract_constants(file_path)
        
        file_analysis[rel_path] = {
            'imports': imports,
            'constants': constants
        }
        
        # Track which files import which modules
        for imp in imports:
            module = imp.get('module', '')
            if module not in all_imports:
                all_imports[module] = []
            all_imports[module].append(rel_path)
            
        # Track which files define which constants
        for const in constants:
            name = const.get('name', '')
            if name not in all_constants:
                all_constants[name] = []
            all_constants[name].append({
                'file': rel_path,
                'value': const.get('value')
            })
    
    # Generate the reports
    os.makedirs(outdir, exist_ok=True)
    
    # Save the full analysis
    with open(os.path.join(outdir, 'file_analysis.json'), 'w') as f:
        json.dump(file_analysis, f, indent=4)
    
    # Save the imports summary
    with open(os.path.join(outdir, 'imports_summary.json'), 'w') as f:
        json.dump(all_imports, f, indent=4)
    
    # Save the constants summary
    with open(os.path.join(outdir, 'constants_summary.json'), 'w') as f:
        json.dump(all_constants, f, indent=4)
    
    # Generate a HTML report for constants analysis
    with open(os.path.join(outdir, 'constants_analysis.html'), 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NHL Playoff Model - Constants Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                th { background-color: #4CAF50; color: white; }
                .inconsistent { background-color: #ffcccc; }
                .consistent { background-color: #ccffcc; }
                .details { margin-left: 20px; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <h1>NHL Playoff Model - Constants Analysis</h1>
            <p>Generated on: """)
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write("""</p>
            
            <h2>Constants Analysis</h2>
            <p>This report shows constants defined across multiple files, highlighting potential inconsistencies.</p>
            
            <table>
                <tr>
                    <th>Constant Name</th>
                    <th>Status</th>
                    <th>Occurrences</th>
                    <th>Values</th>
                </tr>
        """)
        
        # Sort constants by frequency
        for const_name, occurrences in sorted(all_constants.items(), key=lambda x: len(x[1]), reverse=True):
            # Check if values are consistent
            values = [occ.get('value') for occ in occurrences]
            unique_values = set(values)
            
            if len(unique_values) > 1:
                status_class = "inconsistent"
                status_text = "INCONSISTENT"
            else:
                status_class = "consistent"
                status_text = "Consistent"
            
            f.write(f"""
                <tr>
                    <td>{const_name}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{len(occurrences)}</td>
                    <td>
            """)
            
            # List the unique values and their files
            for value in unique_values:
                files = [occ.get('file') for occ in occurrences if occ.get('value') == value]
                f.write(f"<div><strong>Value:</strong> {value}</div>")
                f.write("<div class='details'>")
                for file in files:
                    f.write(f"<div>{file}</div>")
                f.write("</div>")
            
            f.write("""
                    </td>
                </tr>
            """)
        
        f.write("""
            </table>
            
            <h2>Import Analysis</h2>
            <p>This report shows which modules are imported across the codebase.</p>
            
            <table>
                <tr>
                    <th>Module</th>
                    <th>Imported by</th>
                    <th>Import Count</th>
                </tr>
        """)
        
        # Sort modules by import frequency
        for module, files in sorted(all_imports.items(), key=lambda x: len(x[1]), reverse=True):
            if module.startswith('streamlit_app.') or module == 'streamlit_app':
                status_class = "local"
            else:
                status_class = ""
                
            f.write(f"""
                <tr class="{status_class}">
                    <td>{module}</td>
                    <td>
            """)
            
            for file in files:
                f.write(f"<div>{file}</div>")
            
            f.write(f"""
                    </td>
                    <td>{len(files)}</td>
                </tr>
            """)
        
        f.write("""
            </table>
        </body>
        </html>
        """)
    
    print(f"Reports generated in {outdir}")
    return outdir

if __name__ == "__main__":
    # Get command line arguments
    if len(sys.argv) > 2:
        report_dir = sys.argv[2]
    else:
        # Create analysis output directory if it doesn't exist
        tools_dir = str(Path(__file__).parent)
        report_dir = os.path.join(tools_dir, 'analysis_output', datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate the report
    generate_report(report_dir)
    
    print(f"Analysis complete. Reports available in {report_dir}")
