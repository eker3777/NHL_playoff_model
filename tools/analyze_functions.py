"""
Utility script to analyze function definitions across the codebase.
This helps identify duplicated or inconsistent function implementations.
"""

import os
import re
import sys
from pathlib import Path
import ast
import json
from datetime import datetime
import hashlib

def find_python_files(root_dir):
    """Find all Python files in the project."""
    python_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                # Skip analysis scripts
                if not full_path.endswith('analyze_functions.py') and not full_path.endswith('analyze_imports.py'):
                    python_files.append(full_path)
    return python_files

def get_function_hash(node):
    """Generate a hash of the function body to identify similar implementations."""
    function_text = ast.unparse(node)
    # Remove comments and whitespace to focus on functionality
    function_text = re.sub(r'#.*$', '', function_text, flags=re.MULTILINE)
    function_text = re.sub(r'\s+', ' ', function_text)
    return hashlib.md5(function_text.encode()).hexdigest()

def get_function_signature(node):
    """Get a string representation of the function signature."""
    args = []
    
    # Handle positional args
    for arg in node.args.args:
        # Get default value if exists
        args.append(arg.arg)
    
    # Handle keyword args
    for arg in node.args.kwonlyargs:
        args.append(f"{arg.arg}")
    
    # Handle varargs
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    
    # Handle kwargs
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    
    return f"{node.name}({', '.join(args)})"

def extract_functions(file_path):
    """Extract all function definitions from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    functions = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function signature
                signature = get_function_signature(node)
                
                # Get body hash
                body_hash = get_function_hash(node)
                
                # Calculate stats
                lines_of_code = node.end_lineno - node.lineno
                docstring = ast.get_docstring(node)
                has_docstring = docstring is not None
                
                functions.append({
                    'name': node.name,
                    'signature': signature,
                    'body_hash': body_hash,
                    'lines_of_code': lines_of_code,
                    'has_docstring': has_docstring,
                    'lineno': node.lineno,
                    'end_lineno': node.end_lineno
                })
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return []
        
    return functions

def generate_report(outdir):
    """Generate a comprehensive report of function definitions across files."""
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
    all_functions = {}
    function_hashes = {}
    
    for file_path in files:
        rel_path = os.path.relpath(file_path, root_dir)
        functions = extract_functions(file_path)
        
        file_analysis[rel_path] = {
            'functions': functions
        }
        
        # Track which files define which functions
        for func in functions:
            name = func.get('name', '')
            if name not in all_functions:
                all_functions[name] = []
            
            func_info = {
                'file': rel_path,
                'signature': func.get('signature'),
                'body_hash': func.get('body_hash'),
                'lines_of_code': func.get('lines_of_code'),
                'has_docstring': func.get('has_docstring')
            }
            all_functions[name].append(func_info)
            
            # Track functions by hash to find duplicated implementations
            body_hash = func.get('body_hash')
            if body_hash not in function_hashes:
                function_hashes[body_hash] = []
            function_hashes[body_hash].append({
                'name': name,
                'file': rel_path,
                'signature': func.get('signature')
            })
    
    # Ensure the output directory exists
    os.makedirs(outdir, exist_ok=True)
    
    # Save the full analysis
    with open(os.path.join(outdir, 'function_analysis.json'), 'w') as f:
        json.dump(file_analysis, f, indent=4)
    
    # Save the functions summary
    with open(os.path.join(outdir, 'functions_summary.json'), 'w') as f:
        json.dump(all_functions, f, indent=4)
    
    # Generate a markdown report for functions analysis
    with open(os.path.join(outdir, 'function_report.md'), 'w') as f:
        f.write(f"# NHL Playoff Model - Function Analysis\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Find duplicated function implementations
        f.write("## Duplicated Function Implementations\n\n")
        
        duplicated_hashes = {h: funcs for h, funcs in function_hashes.items() if len(funcs) > 1}
        if duplicated_hashes:
            for hash_value, functions in duplicated_hashes.items():
                # Skip very small functions (likely trivial)
                if any(func['name'].startswith('__') for func in functions):
                    continue
                    
                # Skip if the hash is empty
                if hash_value == "d41d8cd98f00b204e9800998ecf8427e":  # MD5 of empty string
                    continue
                
                f.write(f"### {functions[0]['name']} ({len(functions)} occurrences)\n\n")
                f.write("Found in files:\n\n")
                for func in functions:
                    f.write(f"- `{func['file']}` with signature `{func['signature']}`\n")
                f.write("\n")
        else:
            f.write("No duplicated function implementations found.\n\n")
        
        # List functions defined in multiple files
        f.write("## Functions Defined in Multiple Files\n\n")
        multi_defined = {name: funcs for name, funcs in all_functions.items() if len(funcs) > 1}
        
        if multi_defined:
            for name, occurrences in sorted(multi_defined.items(), key=lambda x: len(x[1]), reverse=True):
                # Skip special methods
                if name.startswith('__') and name.endswith('__'):
                    continue
                    
                f.write(f"### {name} ({len(occurrences)} occurrences)\n\n")
                
                # Check if implementations are identical
                body_hashes = [func['body_hash'] for func in occurrences]
                if len(set(body_hashes)) == 1:
                    f.write("*All implementations are identical*\n\n")
                else:
                    f.write("**⚠️ Different implementations found! ⚠️**\n\n")
                
                f.write("Defined in files:\n\n")
                for func in occurrences:
                    doc_status = "✅" if func['has_docstring'] else "❌"
                    f.write(f"- `{func['file']}` ({func['lines_of_code']} lines) [Docstring: {doc_status}]\n")
                f.write("\n")
        else:
            f.write("No functions are defined in multiple files.\n\n")
        
        # List the most complex functions
        f.write("## Most Complex Functions\n\n")
        all_func_list = []
        for name, funcs in all_functions.items():
            for func in funcs:
                all_func_list.append({
                    'name': name,
                    'file': func['file'],
                    'lines': func['lines_of_code'],
                    'has_docstring': func['has_docstring']
                })
        
        # Sort by lines of code
        all_func_list.sort(key=lambda x: x['lines'], reverse=True)
        
        f.write("| Function | File | Lines | Docstring |\n")
        f.write("|----------|------|-------|----------|\n")
        
        for func in all_func_list[:20]:  # Top 20 most complex functions
            doc_status = "✅" if func['has_docstring'] else "❌"
            f.write(f"| {func['name']} | {func['file']} | {func['lines']} | {doc_status} |\n")
        
        f.write("\n")
    
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
