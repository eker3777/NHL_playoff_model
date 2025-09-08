#!/usr/bin/env python3
"""
Quick start guide for the harmoniously merged NHL Playoff Model.

This script demonstrates how to use the improved codebase structure.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and show the output."""
    print(f"\n🔧 {description}")
    print(f"Running: {' '.join(command)}")
    print("-" * 50)

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Demonstrate the new capabilities."""

    print("🏒 NHL Playoff Model - Harmonious Merge Quick Start")
    print("=" * 60)

    # 1. Show project structure
    print("\n📁 Project Structure:")
    structure_command = ["find", ".", "-type", "f", "-name", "*.py", "|", "head", "-20"]
    run_command(["find", ".", "-maxdepth", "2", "-type", "d"], "Project directories")

    # 2. Run tests
    print("\n🧪 Running Tests:")
    if not run_command(
        [sys.executable, "-m", "pytest", "tests/unit/", "-v"], "Unit tests"
    ):
        print("Note: Tests may need additional dependencies to run fully")

    # 3. Check code quality
    print("\n✨ Code Quality Check:")
    files_to_check = ["logging_config.py", "demo_improved_structure.py"]
    for file in files_to_check:
        if Path(file).exists():
            run_command(
                [sys.executable, "-m", "black", "--check", file],
                f"Black formatting check on {file}",
            )

    # 4. Show logging demo
    print("\n📝 Logging System Demo:")
    run_command(
        [sys.executable, "demo_improved_structure.py"], "Demonstrate new logging system"
    )

    # 5. Show available tools
    print("\n🛠️ Available Analysis Tools:")
    tools_dir = Path("tools")
    if tools_dir.exists():
        python_tools = list(tools_dir.glob("*.py"))
        for tool in python_tools[:5]:  # Show first 5 tools
            print(f"  - {tool.name}")
        if len(python_tools) > 5:
            print(f"  ... and {len(python_tools) - 5} more tools")

    # 6. Show documentation
    print("\n📚 Key Documentation:")
    docs = [
        (
            "CODEBASE_REVIEW.md",
            "Implementation guidelines and code quality recommendations",
        ),
        ("IMPLEMENTATION_ROADMAP.md", "Development roadmap with phases"),
        ("MERGE_SUMMARY.md", "Summary of the harmonious merge"),
    ]

    for doc, description in docs:
        if Path(doc).exists():
            print(f"  - {doc}: {description}")

    print("\n" + "=" * 60)
    print("✅ Quick start completed!")
    print("\n📋 Next Steps:")
    print("  1. Review CODEBASE_REVIEW.md for implementation guidelines")
    print("  2. Check IMPLEMENTATION_ROADMAP.md for Phase 2 improvements")
    print("  3. Run tests with: python -m pytest tests/ -v")
    print("  4. Use logging_config.setup_logging() in your scripts")
    print("  5. Apply code quality with: black . && isort .")
    print("\n🚀 Ready for continued development following the roadmap!")


if __name__ == "__main__":
    main()
