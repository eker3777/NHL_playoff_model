"""
Dependency Analysis Tool for NHL Playoff Model

This script analyzes Python files to extract dependencies, import patterns, and 
constant usage patterns to identify inconsistencies in the codebase.
"""

import os
import sys
import ast
import networkx as nx
import re
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime

class DependencyAnalyzer:
    def __init__(self, project_dir, output_dir=None):
        """Initialize the analyzer with project directory"""
        self.project_dir = project_dir
        self.output_dir = output_dir or os.path.join(project_dir, 'analysis_output')
        self.files = {}  # All Python files keyed by path
        self.import_graph = nx.DiGraph()  # Directed graph for imports
        self.function_calls = defaultdict(list)  # Track function calls across files
        self.constants_usage = defaultdict(list)  # Track constants usage
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def find_python_files(self):
        """Find all Python files in the project directory"""
        python_files = []
        for root, _, files in os.walk(self.project_dir):
            # Skip our analysis output directory
            if os.path.abspath(root) == os.path.abspath(self.output_dir):
                continue
                
            # Skip hidden directories
            if os.path.basename(root).startswith('.'):
                continue
            
            # Skip virtual environment directories
            if '.venv' in root or 'venv' in root or '__pycache__' in root:
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_dir)
                    python_files.append((rel_path, file_path))
        
        self.files = dict(python_files)
        print(f"Found {len(self.files)} Python files in {self.project_dir}")
        return self.files
    
    def analyze_file(self, file_path):
        """Analyze a single Python file for imports, constants, and function calls"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            file_info = {
                'imports': [],
                'from_imports': [],
                'constants': [],
                'functions': [],
                'function_calls': [],
                'modules_called': []  # Changed from set to list for JSON serialization
            }
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        file_info['imports'].append(name.name)
                        # Add edge to import graph
                        self.import_graph.add_edge(file_path, name.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for name in node.names:
                            import_from = f"{node.module}.{name.name}"
                            file_info['from_imports'].append(import_from)
                            # Add edge to import graph
                            self.import_graph.add_edge(file_path, node.module)
                
                # Extract constants
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            name = target.id
                            if name.isupper():  # Assume constants are UPPERCASE
                                file_info['constants'].append(name)
                
                # Extract function definitions
                elif isinstance(node, ast.FunctionDef):
                    file_info['functions'].append(node.name)
                
                # Extract function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        file_info['function_calls'].append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            # This is a module.function() call
                            module_name = node.func.value.id
                            function_name = node.func.attr
                            file_info['function_calls'].append(f"{module_name}.{function_name}")
                            if module_name not in file_info['modules_called']:
                                file_info['modules_called'].append(module_name)
            
            return file_info
        
        except Exception as e:
            print(f"Error analyzing file {file_path}: {str(e)}")
            return None
    
    def analyze_all_files(self):
        """Analyze all Python files in the project"""
        results = {}
        for rel_path, abs_path in self.files.items():
            file_info = self.analyze_file(abs_path)
            if file_info:
                results[rel_path] = file_info
        
        # Save results to JSON - ensuring all data is JSON serializable
        results_path = os.path.join(self.output_dir, 'file_analysis.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        
        print(f"Analysis results saved to {results_path}")
        return results
    
    def _json_serializer(self, obj):
        """Custom JSON serializer to handle non-serializable objects"""
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def analyze_constants_usage(self, results):
        """Analyze how constants are used across files"""
        # Look for key constants we want to track
        key_constants = ['HOME_ICE_ADVANTAGE', 'SERIES_LENGTH_DIST', 'SERIES_LENGTH_DISTRIBUTION', 
                         'API_TIMEOUT', 'API_BASE_URL', 'CRITICAL_FEATURES']
        
        constants_by_file = {}
        constant_sources = {}
        
        # First find all constants defined in each file
        for file_path, info in results.items():
            constants_by_file[file_path] = info.get('constants', [])
            
            # Track which file defines which constants
            for constant in info.get('constants', []):
                if constant not in constant_sources:
                    constant_sources[constant] = []
                constant_sources[constant].append(file_path)
        
        # Look for key constants defined in multiple files
        duplicate_constants = {}
        for constant, sources in constant_sources.items():
            if len(sources) > 1 and constant in key_constants:
                duplicate_constants[constant] = sources
        
        # Look for hardcoded values that should be constants
        hardcoded_values = []
        # (This would require more sophisticated source code analysis)
        
        # Save constants analysis to JSON
        constants_analysis = {
            'constants_by_file': constants_by_file,
            'duplicate_constants': duplicate_constants,
            'hardcoded_values': hardcoded_values
        }
        
        constants_path = os.path.join(self.output_dir, 'constants_analysis.json')
        with open(constants_path, 'w') as f:
            json.dump(constants_analysis, f, indent=2, default=self._json_serializer)
        
        print(f"Constants analysis saved to {constants_path}")
        return constants_analysis
    
    def analyze_function_usage(self, results):
        """Analyze how functions are used across files"""
        # Key functions to track
        key_functions = [
            'standardize_percentage',
            'format_percentage_for_display',
            'validate_and_fix',
            'predict_series_winner',
            'simulate_playoff_bracket',
            'get_series_schedule'
        ]
        
        function_definitions = {}
        function_calls_by_file = {}
        function_usage = defaultdict(list)
        
        # Track function definitions and calls
        for file_path, info in results.items():
            # Track where functions are defined
            for func in info.get('functions', []):
                if func not in function_definitions:
                    function_definitions[func] = []
                function_definitions[func].append(file_path)
            
            # Track function calls
            function_calls_by_file[file_path] = info.get('function_calls', [])
            
            # Track usage of key functions
            for func_call in info.get('function_calls', []):
                # Handle both direct calls and module.function() calls
                func_name = func_call.split('.')[-1] if '.' in func_call else func_call
                if func_name in key_functions:
                    function_usage[func_name].append(file_path)
        
        # Find duplicate function definitions
        duplicate_functions = {}
        for func, sources in function_definitions.items():
            if len(sources) > 1:
                duplicate_functions[func] = sources
        
        # Save function analysis to JSON
        function_analysis = {
            'function_definitions': function_definitions,
            'function_calls_by_file': function_calls_by_file,
            'key_function_usage': dict(function_usage),
            'duplicate_functions': duplicate_functions
        }
        
        function_path = os.path.join(self.output_dir, 'function_analysis.json')
        with open(function_path, 'w') as f:
            json.dump(function_analysis, f, indent=2, default=self._json_serializer)
        
        print(f"Function analysis saved to {function_path}")
        return function_analysis
    
    def generate_dependency_graph(self):
        """Generate a visualization of the dependency graph"""
        # Create a directed graph of file dependencies
        dep_graph = nx.DiGraph()
        
        # Add nodes for all files
        for rel_path in self.files.keys():
            # Use shortened labels for clarity
            label = os.path.basename(rel_path)
            dep_graph.add_node(rel_path, label=label)
        
        # Add edges based on imports
        for source, target in self.import_graph.edges():
            # Only add edges between files in our project
            if source in self.files.values() and target in self.files.values():
                source_rel = next(k for k, v in self.files.items() if v == source)
                target_rel = next(k for k, v in self.files.items() if v == target)
                dep_graph.add_edge(source_rel, target_rel)
        
        # Save the graph as a dot file for visualization with graphviz
        graph_path = os.path.join(self.output_dir, 'dependency_graph.dot')
        nx.drawing.nx_pydot.write_dot(dep_graph, graph_path)
        
        # Also try to create a PNG if pydot is available
        try:
            import pydot
            graph_image_path = os.path.join(self.output_dir, 'dependency_graph.png')
            (pydot.graph_from_dot_file(graph_path)[0]).write_png(graph_image_path)
            print(f"Dependency graph image saved to {graph_image_path}")
        except Exception as e:
            print(f"Could not create PNG image of graph: {str(e)}")
        
        print(f"Dependency graph saved to {graph_path}")
        return dep_graph
    
    def generate_report(self, results, constants_analysis, function_analysis):
        """Generate a summary report of the analysis"""
        report = [
            "# NHL Playoff Model - Cross-Functionality Analysis Report",
            "",
            f"Analysis performed on {len(self.files)} Python files.",
            "",
            "## Key Findings",
            ""
        ]
        
        # Constants Findings
        report.append("### Constants Analysis")
        report.append("")
        
        if constants_analysis['duplicate_constants']:
            report.append("#### Duplicate Constants")
            report.append("")
            for constant, sources in constants_analysis['duplicate_constants'].items():
                report.append(f"- `{constant}` is defined in multiple files: {', '.join(sources)}")
            report.append("")
        else:
            report.append("No duplicate constants found.")
            report.append("")
        
        # Function Usage Findings
        report.append("### Function Usage Analysis")
        report.append("")
        
        if function_analysis['duplicate_functions']:
            report.append("#### Duplicate Function Definitions")
            report.append("")
            for func, sources in function_analysis['duplicate_functions'].items():
                report.append(f"- `{func}()` is defined in multiple files: {', '.join(sources)}")
            report.append("")
        else:
            report.append("No duplicate function definitions found.")
            report.append("")
        
        report.append("#### Key Function Usage")
        report.append("")
        for func, files in function_analysis['key_function_usage'].items():
            report.append(f"- `{func}()` is used in {len(files)} files")
        report.append("")
        
        # Module Import Analysis
        report.append("### Import Analysis")
        report.append("")
        
        # Count types of imports by file
        import_counts = {}
        for file_path, info in results.items():
            import_counts[file_path] = {
                'standard_lib': 0,
                'third_party': 0,
                'local': 0
            }
            
            for imp in info.get('imports', []):
                if imp in ['os', 'sys', 'datetime', 'json', 're', 'math', 'time', 'random']:
                    import_counts[file_path]['standard_lib'] += 1
                elif imp in ['streamlit', 'pandas', 'numpy', 'matplotlib', 'sklearn']:
                    import_counts[file_path]['third_party'] += 1
                else:
                    import_counts[file_path]['local'] += 1
            
            for imp in info.get('from_imports', []):
                if imp.split('.')[0] in ['os', 'sys', 'datetime', 'json', 're', 'math', 'time', 'random']:
                    import_counts[file_path]['standard_lib'] += 1
                elif imp.split('.')[0] in ['streamlit', 'pandas', 'numpy', 'matplotlib', 'sklearn']:
                    import_counts[file_path]['third_party'] += 1
                else:
                    import_counts[file_path]['local'] += 1
        
        # Report on files with the most complex import patterns
        complex_imports = sorted(import_counts.items(), 
                               key=lambda x: sum(x[1].values()), 
                               reverse=True)[:5]
        
        report.append("#### Most Complex Import Patterns")
        report.append("")
        for file_path, counts in complex_imports:
            total = sum(counts.values())
            report.append(f"- {file_path}: {total} total imports ({counts['standard_lib']} stdlib, "
                        f"{counts['third_party']} third-party, {counts['local']} local)")
        report.append("")
        
        # Save the report
        report_path = os.path.join(self.output_dir, 'analysis_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Analysis report saved to {report_path}")
        return report
    
    def run_analysis(self):
        """Run the full analysis pipeline"""
        self.find_python_files()
        results = self.analyze_all_files()
        constants_analysis = self.analyze_constants_usage(results)
        function_analysis = self.analyze_function_usage(results)
        self.generate_dependency_graph()
        self.generate_report(results, constants_analysis, function_analysis)
        
        return {
            'file_analysis': results,
            'constants_analysis': constants_analysis,
            'function_analysis': function_analysis
        }

if __name__ == "__main__":
    # Get the project directory from command-line arguments
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Get the output directory from command-line arguments
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = os.path.join(project_dir, 'analysis_output', datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Create analyzer with the specified directories
    analyzer = DependencyAnalyzer(project_dir, output_dir)
    analyzer.run_analysis()
