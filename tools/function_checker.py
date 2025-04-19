"""
Function Consistency Checker for NHL Playoff Model

This script analyzes function usage across the codebase to ensure consistent
function signatures, parameter names, and return values.
"""

import os
import ast
import inspect
import json
from collections import defaultdict

class FunctionChecker:
    def __init__(self, project_dir, output_dir=None):
        """Initialize the checker with project directory"""
        self.project_dir = project_dir
        self.output_dir = output_dir or os.path.join(project_dir, 'analysis_output')
        self.files = {}
        self.key_functions = [
            'standardize_percentage',
            'format_percentage_for_display',
            'validate_and_fix',
            'predict_series_winner',
            'simulate_playoff_bracket',
            'get_series_schedule',
            'load_team_data',
            'load_models',
            'run_playoff_simulations'
        ]
        self.function_definitions = defaultdict(list)
        self.function_calls = defaultdict(list)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
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
    
    def analyze_function_definitions(self, file_path):
        """Analyze function definitions in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            functions = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get function name
                    func_name = node.name
                    
                    # Get parameters
                    params = []
                    defaults = {}
                    
                    # Handle positional args
                    for arg in node.args.args:
                        # Support for different Python versions
                        if hasattr(arg, 'arg'):
                            params.append(arg.arg)
                        else:
                            params.append(arg.id)
                    
                    # Handle defaults for positional args
                    if node.args.defaults:
                        for i, default in enumerate(node.args.defaults):
                            default_idx = len(node.args.args) - len(node.args.defaults) + i
                            param_name = node.args.args[default_idx].arg if hasattr(node.args.args[default_idx], 'arg') else node.args.args[default_idx].id
                            
                            if isinstance(default, ast.Num):
                                defaults[param_name] = default.n
                            elif isinstance(default, ast.Str):
                                defaults[param_name] = default.s
                            elif isinstance(default, ast.NameConstant):
                                defaults[param_name] = default.value
                            elif hasattr(ast, 'Constant') and isinstance(default, ast.Constant):
                                defaults[param_name] = default.value
                            elif isinstance(default, ast.List):
                                defaults[param_name] = "list"
                            elif isinstance(default, ast.Dict):
                                defaults[param_name] = "dict"
                            else:
                                defaults[param_name] = str(type(default))
                    
                    # Handle *args
                    if node.args.vararg:
                        vararg_name = node.args.vararg.arg if hasattr(node.args.vararg, 'arg') else node.args.vararg
                        params.append(f"*{vararg_name}")
                    
                    # Handle keyword-only args
                    if hasattr(node.args, 'kwonlyargs'):
                        for arg in node.args.kwonlyargs:
                            arg_name = arg.arg if hasattr(arg, 'arg') else arg.id
                            params.append(arg_name)
                    
                    # Handle defaults for keyword-only args
                    if hasattr(node.args, 'kw_defaults'):
                        for i, default in enumerate(node.args.kw_defaults):
                            if default:  # Could be None
                                param_name = node.args.kwonlyargs[i].arg if hasattr(node.args.kwonlyargs[i], 'arg') else node.args.kwonlyargs[i].id
                                
                                if isinstance(default, ast.Num):
                                    defaults[param_name] = default.n
                                elif isinstance(default, ast.Str):
                                    defaults[param_name] = default.s
                                elif isinstance(default, ast.NameConstant):
                                    defaults[param_name] = default.value
                                elif hasattr(ast, 'Constant') and isinstance(default, ast.Constant):
                                    defaults[param_name] = default.value
                                elif isinstance(default, ast.List):
                                    defaults[param_name] = "list"
                                elif isinstance(default, ast.Dict):
                                    defaults[param_name] = "dict"
                                else:
                                    defaults[param_name] = str(type(default))
                    
                    # Handle **kwargs
                    if node.args.kwarg:
                        kwarg_name = node.args.kwarg.arg if hasattr(node.args.kwarg, 'arg') else node.args.kwarg
                        params.append(f"**{kwarg_name}")
                    
                    # Extract docstring if available
                    docstring = ast.get_docstring(node)
                    
                    # Store the function definition
                    functions[func_name] = {
                        'params': params,
                        'defaults': defaults,
                        'docstring': docstring,
                        'line_number': node.lineno,
                        'end_line': node.end_lineno if hasattr(node, 'end_lineno') else 0
                    }
                    
                    # Update global tracking
                    self.function_definitions[func_name].append({
                        'file': file_path,
                        'signature': params,
                        'defaults': defaults
                    })
            
            return functions
        
        except Exception as e:
            print(f"Error analyzing functions in {file_path}: {str(e)}")
            return {}
    
    def analyze_function_calls(self, file_path):
        """Analyze function calls in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            function_calls = defaultdict(list)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Extract function name
                    if isinstance(node.func, ast.Name):
                        # Direct function call
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        # Method call or module.function()
                        func_name = node.func.attr
                        if isinstance(node.func.value, ast.Name):
                            # This is a module.function() call
                            module_name = node.func.value.id
                            func_name = f"{module_name}.{func_name}"
                    else:
                        continue  # Skip complex calls
                    
                    # Extract arguments
                    args = []
                    for arg in node.args:
                        if isinstance(arg, ast.Name):
                            args.append(arg.id)
                        elif hasattr(ast, 'Constant') and isinstance(arg, ast.Constant):
                            args.append(repr(arg.value))
                        elif isinstance(arg, (ast.Str, ast.Num, ast.NameConstant)):
                            if hasattr(arg, 's'):  # Str
                                args.append(repr(arg.s))
                            elif hasattr(arg, 'n'):  # Num
                                args.append(repr(arg.n))
                            elif hasattr(arg, 'value'):  # NameConstant
                                args.append(repr(arg.value))
                            else:
                                args.append("<complex-arg>")
                        else:
                            args.append("<complex-arg>")
                    
                    # Extract keyword arguments
                    kwargs = {}
                    for kw in node.keywords:
                        kwargs[kw.arg] = "<value>" if kw.arg else "**<value>"
                    
                    # Update function calls tracking
                    function_calls[func_name].append({
                        'args': args,
                        'kwargs': kwargs,
                        'line_number': node.lineno
                    })
                    
                    # Update global tracking for key functions
                    for key_func in self.key_functions:
                        if key_func in func_name:
                            self.function_calls[key_func].append({
                                'file': file_path,
                                'args': args,
                                'kwargs': kwargs
                            })
            
            return dict(function_calls)
        
        except Exception as e:
            print(f"Error analyzing function calls in {file_path}: {str(e)}")
            return {}
    
    def _json_serializer(self, obj):
        """Custom JSON serializer to handle non-serializable objects"""
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (tuple, bytes)):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def analyze_all_files(self):
        """Analyze all Python files for function definitions and calls"""
        results = {}
        for rel_path, abs_path in self.files.items():
            try:
                file_functions = self.analyze_function_definitions(abs_path)
                file_calls = self.analyze_function_calls(abs_path)
                
                results[rel_path] = {
                    'functions': file_functions,
                    'calls': file_calls
                }
            except Exception as e:
                print(f"Error analyzing file {rel_path}: {str(e)}")
        
        # Save results to JSON
        results_path = os.path.join(self.output_dir, 'function_analysis.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        
        print(f"Function analysis saved to {results_path}")
        return results
    
    def check_function_consistency(self):
        """Check for consistency in function signatures across files"""
        # Check for functions defined in multiple files
        multiple_definitions = {}
        for func_name, definitions in self.function_definitions.items():
            if len(definitions) > 1:
                multiple_definitions[func_name] = definitions
        
        # Check parameter consistency for functions defined in multiple places
        parameter_inconsistencies = {}
        for func_name, definitions in multiple_definitions.items():
            unique_signatures = set()
            for definition in definitions:
                signature_str = ','.join(definition['signature'])
                unique_signatures.add(signature_str)
                
            if len(unique_signatures) > 1:
                parameter_inconsistencies[func_name] = {
                    'definitions': definitions,
                    'unique_signatures': list(unique_signatures)
                }
        
        # Check for common key functions with different calling patterns
        call_pattern_inconsistencies = {}
        for func_name, calls in self.function_calls.items():
            if len(calls) > 1:
                # Check for variations in kwargs usage
                kwarg_keys = set()
                for call in calls:
                    kwarg_keys.update(call.get('kwargs', {}).keys())
                
                # If there are a lot of different kwargs being used, this might indicate inconsistency
                if len(kwarg_keys) > 3:
                    call_pattern_inconsistencies[func_name] = {
                        'calls': calls,
                        'kwarg_variations': list(kwarg_keys)
                    }
        
        return {
            'multiple_definitions': multiple_definitions,
            'parameter_inconsistencies': parameter_inconsistencies,
            'call_pattern_inconsistencies': call_pattern_inconsistencies
        }
    
    def generate_function_report(self, consistency_checks):
        """Generate a report on function consistency"""
        report = [
            "# NHL Playoff Model - Function Consistency Report",
            "",
            "## Overview",
            "",
            f"Analysis performed on {len(self.files)} Python files.",
            "",
            "## Key Function Analysis",
            ""
        ]
        
        # Report on key functions
        for func_name in self.key_functions:
            definitions = self.function_definitions.get(func_name, [])
            calls = self.function_calls.get(func_name, [])
            
            report.append(f"### `{func_name}()`")
            report.append("")
            
            if definitions:
                report.append("#### Defined in:")
                report.append("")
                for definition in definitions:
                    signature = ', '.join(definition['signature'])
                    report.append(f"- {definition['file']}: `{func_name}({signature})`")
                report.append("")
            else:
                report.append("Not defined in any analyzed file.")
                report.append("")
            
            if calls:
                report.append("#### Called from:")
                report.append("")
                for i, call in enumerate(calls):
                    if i >= 10:  # Limit to 10 examples to keep report manageable
                        report.append(f"- ... and {len(calls) - 10} more calls")
                        break
                    args_str = ', '.join(call['args'])
                    kwargs_str = ', '.join([f"{k}={v}" for k, v in call.get('kwargs', {}).items()])
                    all_args = args_str + (', ' if args_str and kwargs_str else '') + kwargs_str
                    report.append(f"- {call['file']}: `{func_name}({all_args})`")
                report.append("")
            else:
                report.append("Not called in any analyzed file.")
                report.append("")
        
        # Report on functions defined in multiple places
        multiple_defs = consistency_checks.get('multiple_definitions', {})
        if multiple_defs:
            report.append("## Functions Defined in Multiple Files")
            report.append("")
            for func_name, definitions in multiple_defs.items():
                if func_name in self.key_functions:
                    continue  # Already covered above
                    
                report.append(f"### `{func_name}()`")
                report.append("")
                for definition in definitions:
                    signature = ', '.join(definition['signature'])
                    report.append(f"- {definition['file']}: `{func_name}({signature})`")
                report.append("")
        
        # Report on parameter inconsistencies
        param_inconsistencies = consistency_checks.get('parameter_inconsistencies', {})
        if param_inconsistencies:
            report.append("## Functions with Inconsistent Parameters")
            report.append("")
            for func_name, details in param_inconsistencies.items():
                report.append(f"### `{func_name}()`")
                report.append("")
                report.append("Different signatures:")
                report.append("")
                for signature in details['unique_signatures']:
                    report.append(f"- `{func_name}({signature})`")
                report.append("")
                report.append("Defined in:")
                report.append("")
                for definition in details['definitions']:
                    report.append(f"- {definition['file']}")
                report.append("")
        
        # Report on call pattern inconsistencies
        call_inconsistencies = consistency_checks.get('call_pattern_inconsistencies', {})
        if call_inconsistencies:
            report.append("## Functions with Inconsistent Call Patterns")
            report.append("")
            for func_name, details in call_inconsistencies.items():
                report.append(f"### `{func_name}()`")
                report.append("")
                report.append("Keyword argument variations:")
                report.append("")
                for kwarg in details['kwarg_variations']:
                    report.append(f"- `{kwarg}`")
                report.append("")
                report.append("Called from:")
                report.append("")
                unique_files = set()
                for call in details['calls']:
                    unique_files.add(call['file'])
                for file in sorted(unique_files):
                    report.append(f"- {file}")
                report.append("")
        
        # Save the report
        report_path = os.path.join(self.output_dir, 'function_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Function consistency report saved to {report_path}")
        return report
    
    def run_checks(self):
        """Run the full function consistency checking process"""
        self.find_python_files()
        analysis_results = self.analyze_all_files()
        consistency_checks = self.check_function_consistency()
        self.generate_function_report(consistency_checks)
        
        return {
            'analysis': analysis_results,
            'consistency_checks': consistency_checks
        }

if __name__ == "__main__":
    # Set the project directory to the current directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checker = FunctionChecker(project_dir)
    checker.run_checks()
