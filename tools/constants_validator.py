"""
Constants Usage Validator for NHL Playoff Model

This script specifically focuses on analyzing how constants are used across the codebase,
identifying hardcoded values that should be in config.py, and ensuring consistent usage
of critical constants like HOME_ICE_ADVANTAGE and SERIES_LENGTH_DIST.
"""

import os
import re
import sys
import ast
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class ConstantsValidator:
    def __init__(self, project_dir, output_dir=None):
        """Initialize the validator with project directory"""
        self.project_dir = project_dir
        self.output_dir = output_dir or os.path.join(project_dir, 'analysis_output')
        self.files = {}
        self.critical_constants = {
            'HOME_ICE_ADVANTAGE': [0.039, 0.04, 3.9],  # Values to look for (decimal and percentage)
            'SERIES_LENGTH_DIST': [[0.14, 0.243, 0.336, 0.281]],  # Values to look for
            'SERIES_LENGTH_DISTRIBUTION': [[0.14, 0.243, 0.336, 0.281]],
            'API_TIMEOUT': [30, 60],
            'API_BASE_URL': [],  # Will look for any string URLs
            'REFRESH_HOUR': [5],
            'CRITICAL_FEATURES': []  # Will look for any list definitions
        }
        self.hardcoded_values = defaultdict(list)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def find_python_files(self):
        """Find all Python files in the project directory"""
        python_files = []
        for root, _, files in os.walk(self.project_dir):
            if os.path.abspath(root) == os.path.abspath(self.output_dir):
                continue
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
    
    def _get_constant_value(self, node):
        """Extract constant value from an AST node"""
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.List):
            values = []
            for elt in node.elts:
                value = self._get_constant_value(elt)
                if value is not None:
                    values.append(value)
            return values
        elif isinstance(node, ast.Tuple):
            values = []
            for elt in node.elts:
                value = self._get_constant_value(elt)
                if value is not None:
                    values.append(value)
            return tuple(values)
        elif isinstance(node, ast.Dict):
            result = {}
            for k, v in zip(node.keys, node.values):
                key = self._get_constant_value(k)
                value = self._get_constant_value(v)
                if key is not None and value is not None:
                    result[key] = value
            return result
        elif isinstance(node, ast.NameConstant):
            return node.value
        # Handle AST version differences (Python 3.8+)
        elif hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
            return node.value
        return None
    
    def analyze_file_constants(self, file_path):
        """Analyze constants defined and used in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            results = {
                'defined_constants': {},
                'imported_constants': [],
                'hardcoded_values': []
            }
            
            # Track imports from config.py
            config_imports = []
            
            # Find imports from config
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and 'config' in node.module:
                    for name in node.names:
                        config_imports.append(name.name)
                        results['imported_constants'].append(name.name)
            
            # Find defined constants
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            constant_name = target.id
                            constant_value = self._get_constant_value(node.value)
                            results['defined_constants'][constant_name] = constant_value
                            
                            # Check if this is a critical constant that should be imported
                            if constant_name in self.critical_constants and constant_name not in config_imports:
                                self.hardcoded_values[constant_name].append({
                                    'file': file_path,
                                    'value': constant_value
                                })
            
            # Look for hardcoded values that match our critical constants
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) or isinstance(node, ast.Expr) or isinstance(node, ast.Call):
                    # For floating point literals that match HOME_ICE_ADVANTAGE
                    if isinstance(node, ast.Assign) and hasattr(node.value, 'n'):
                        value = node.value.n
                        if value in self.critical_constants['HOME_ICE_ADVANTAGE']:
                            results['hardcoded_values'].append({
                                'type': 'HOME_ICE_ADVANTAGE',
                                'value': value,
                                'line': node.lineno
                            })
                    
                    # For list literals that match SERIES_LENGTH_DIST
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
                        elements = []
                        for elt in node.value.elts:
                            value = self._get_constant_value(elt)
                            if value is not None:
                                elements.append(value)
                                
                        if len(elements) == 4 and all(isinstance(e, (int, float)) for e in elements):
                            # Check if this is likely to be SERIES_LENGTH_DIST
                            if abs(sum(elements) - 1.0) < 0.01:
                                results['hardcoded_values'].append({
                                    'type': 'SERIES_LENGTH_DIST',
                                    'value': elements,
                                    'line': node.lineno
                                })
            
            return results
        
        except Exception as e:
            print(f"Error analyzing constants in {file_path}: {str(e)}")
            return None
    
    def _json_serializer(self, obj):
        """Custom JSON serializer to handle non-serializable objects"""
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (tuple, bytes)):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def analyze_all_files(self):
        """Analyze all Python files for constants usage"""
        results = {}
        for rel_path, abs_path in self.files.items():
            file_results = self.analyze_file_constants(abs_path)
            if file_results:
                results[rel_path] = file_results
        
        # Save results to JSON
        results_path = os.path.join(self.output_dir, 'constants_usage.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        
        print(f"Constants usage analysis saved to {results_path}")
        return results
    
    def find_config_file(self):
        """Find the config.py file in the project"""
        config_files = []
        for rel_path, abs_path in self.files.items():
            if rel_path.endswith('config.py'):
                config_files.append((rel_path, abs_path))
        
        if not config_files:
            print("Warning: No config.py file found!")
            return None
        
        if len(config_files) > 1:
            print(f"Warning: Multiple config.py files found: {config_files}")
        
        return config_files[0]
    
    def analyze_config_file(self):
        """Analyze the config.py file to identify defined constants"""
        config_file = self.find_config_file()
        if not config_file:
            return {}
        
        rel_path, abs_path = config_file
        results = self.analyze_file_constants(abs_path)
        
        # Save config analysis to JSON
        config_path = os.path.join(self.output_dir, 'config_analysis.json')
        with open(config_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        
        print(f"Config file analysis saved to {config_path}")
        return results
    
    def generate_constants_report(self, all_results, config_results):
        """Generate a report on constants usage consistency"""
        report = [
            "# NHL Playoff Model - Constants Usage Report",
            "",
            "## Overview",
            "",
            f"Analysis performed on {len(self.files)} Python files.",
            "",
            "## Critical Constants Analysis",
            ""
        ]
        
        # Get constants defined in config.py
        config_constants = config_results.get('defined_constants', {})
        report.append(f"Found {len(config_constants)} constants defined in config.py:")
        report.append("")
        
        for const, value in config_constants.items():
            report.append(f"- `{const}` = `{value}`")
        report.append("")
        
        # Files importing from config
        files_importing_config = []
        for rel_path, results in all_results.items():
            imported = results.get('imported_constants', [])
            if imported:
                files_importing_config.append((rel_path, imported))
        
        report.append(f"{len(files_importing_config)} files import constants from config.py:")
        report.append("")
        
        files_not_importing = []
        for rel_path in self.files.keys():
            if rel_path.endswith('config.py'):
                continue
                
            if not any(rel_path == file_path for file_path, _ in files_importing_config):
                files_not_importing.append(rel_path)
        
        if files_not_importing:
            report.append("### Files NOT importing from config.py:")
            report.append("")
            for file_path in files_not_importing:
                report.append(f"- {file_path}")
            report.append("")
        
        # Duplicate definitions of critical constants
        if self.hardcoded_values:
            report.append("### Critical Constants Defined Outside config.py:")
            report.append("")
            for const, instances in self.hardcoded_values.items():
                report.append(f"#### `{const}`")
                report.append("")
                for instance in instances:
                    report.append(f"- Defined in {instance['file']} with value: {instance['value']}")
                report.append("")
        
        # Hardcoded values that should be constants
        hardcoded_by_type = defaultdict(list)
        for rel_path, results in all_results.items():
            for item in results.get('hardcoded_values', []):
                hardcoded_by_type[item['type']].append({
                    'file': rel_path,
                    'value': item['value'],
                    'line': item['line']
                })
        
        if hardcoded_by_type:
            report.append("### Hardcoded Values That Should Use Constants:")
            report.append("")
            for const_type, instances in hardcoded_by_type.items():
                report.append(f"#### `{const_type}`")
                report.append("")
                for instance in instances:
                    report.append(f"- Found in {instance['file']} line {instance['line']}: {instance['value']}")
                report.append("")
        
        # Save the report
        report_path = os.path.join(self.output_dir, 'constants_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Constants usage report saved to {report_path}")
        return report
    
    def run_validation(self):
        """Run the full constants validation process"""
        self.find_python_files()
        all_results = self.analyze_all_files()
        config_results = self.analyze_config_file()
        self.generate_constants_report(all_results, config_results)
        
        return {
            'all_results': all_results,
            'config_results': config_results,
            'hardcoded_values': dict(self.hardcoded_values)
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
    
    # Run the validation with the specified directories
    validator = ConstantsValidator(project_dir, output_dir)
    validator.run_validation()
