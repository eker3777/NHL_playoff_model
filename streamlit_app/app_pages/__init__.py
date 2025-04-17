# Import all page modules to make them available when importing from pages
try:
    from . import first_round
    from . import simulation_results
    from . import head_to_head
    from . import sim_bracket
    from . import about
except ImportError as e:
    # Gracefully handle missing modules during development
    import sys
    print(f"Warning: Not all page modules are available: {e}", file=sys.stderr)
