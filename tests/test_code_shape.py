"""The shape the code keeps (CONTRIBUTING.md): a function fits on a screen."""

import ast
from pathlib import Path

LONGEST = 60


def test_no_function_is_longer_than_a_screen():
    too_long = []
    for path in sorted(Path("recommender").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                lines = node.end_lineno - node.lineno + 1
                if lines > LONGEST:
                    too_long.append(f"{path}:{node.lineno} {node.name} ({lines} lines)")
    assert not too_long, too_long
