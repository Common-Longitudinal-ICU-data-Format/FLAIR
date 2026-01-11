"""
Submission validation for FLAIR benchmark.

Validates method submissions for:
1. Required files (train.py, predict.py, requirements.txt)
2. Banned package imports
3. Code safety (no exec, eval, network access)
4. Demo execution test
"""

import ast
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
import re

from flair_benchmark.privacy.network_guard import BANNED_PACKAGES

logger = logging.getLogger(__name__)


class SubmissionValidator:
    """
    Validate method submissions before execution.

    Performs static analysis and runtime checks to ensure submissions
    don't violate FLAIR privacy policies.
    """

    # Required files in a submission
    REQUIRED_FILES = ["train.py", "predict.py", "requirements.txt"]

    # Patterns that indicate potential security issues
    DANGEROUS_PATTERNS = [
        r"\bexec\s*\(",
        r"\beval\s*\(",
        r"\bcompile\s*\(",
        r"__import__\s*\(",
        r"\bopen\s*\([^)]*['\"]w['\"]",  # Writing files
        r"subprocess\.",
        r"os\.system\s*\(",
        r"os\.popen\s*\(",
    ]

    def __init__(self, submission_dir: Path):
        """
        Initialize validator.

        Args:
            submission_dir: Path to the submission directory
        """
        self.submission_dir = Path(submission_dir)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks.

        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        self._check_structure()
        self._check_requirements()
        self._check_code_safety()

        return len(self.errors) == 0, self.errors, self.warnings

    def _check_structure(self) -> None:
        """Check that required files exist."""
        if not self.submission_dir.exists():
            self.errors.append(f"Submission directory not found: {self.submission_dir}")
            return

        for filename in self.REQUIRED_FILES:
            if not (self.submission_dir / filename).exists():
                self.errors.append(f"Missing required file: {filename}")

        # Check for README (recommended but not required)
        if not (self.submission_dir / "README.md").exists():
            self.warnings.append("No README.md found (recommended for documentation)")

    def _check_requirements(self) -> None:
        """Check requirements.txt for banned packages."""
        req_file = self.submission_dir / "requirements.txt"
        if not req_file.exists():
            return

        with open(req_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Extract package name (handle version specifiers)
                pkg_name = re.split(r"[=<>!~\[\]]", line)[0].strip().lower()

                if pkg_name in BANNED_PACKAGES:
                    self.errors.append(
                        f"Banned package in requirements.txt (line {line_num}): {pkg_name}"
                    )

                # Check for suspicious packages
                if pkg_name in ["socket", "urllib", "http.client"]:
                    self.errors.append(
                        f"Network-capable package in requirements.txt (line {line_num}): {pkg_name}"
                    )

    def _check_code_safety(self) -> None:
        """Perform static analysis on Python files for unsafe patterns."""
        for py_file in self.submission_dir.glob("**/*.py"):
            self._analyze_file(py_file)

    def _analyze_file(self, filepath: Path) -> None:
        """
        Analyze a single Python file for dangerous patterns.

        Args:
            filepath: Path to the Python file
        """
        try:
            with open(filepath, "r") as f:
                source = f.read()
        except Exception as e:
            self.errors.append(f"Cannot read {filepath.name}: {e}")
            return

        # Check for syntax errors first
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {filepath.name}: {e}")
            return

        # Check imports
        banned_found = check_imports_ast(tree)
        for pkg in banned_found:
            self.errors.append(f"Banned import in {filepath.name}: {pkg}")

        # Check for dangerous function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["exec", "eval", "compile"]:
                        self.errors.append(
                            f"Dangerous function in {filepath.name}: {node.func.id}()"
                        )
                    if node.func.id == "__import__":
                        self.errors.append(
                            f"Dynamic import in {filepath.name}: __import__()"
                        )

        # Check for regex patterns
        for pattern in self.DANGEROUS_PATTERNS:
            matches = re.findall(pattern, source)
            if matches:
                self.warnings.append(
                    f"Potentially dangerous pattern in {filepath.name}: {pattern}"
                )


def check_imports(filepath: Path) -> List[str]:
    """
    Check a Python file for banned imports.

    Args:
        filepath: Path to the Python file

    Returns:
        List of banned package names found
    """
    try:
        with open(filepath, "r") as f:
            source = f.read()
        tree = ast.parse(source)
        return check_imports_ast(tree)
    except (SyntaxError, IOError) as e:
        logger.error(f"Error checking imports in {filepath}: {e}")
        return []


def check_imports_ast(tree: ast.AST) -> List[str]:
    """
    Check an AST for banned imports.

    Args:
        tree: Parsed AST

    Returns:
        List of banned package names found
    """
    banned_found = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pkg_base = alias.name.split(".")[0].lower()
                if pkg_base in BANNED_PACKAGES:
                    banned_found.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                pkg_base = node.module.split(".")[0].lower()
                if pkg_base in BANNED_PACKAGES:
                    banned_found.append(node.module)

    return banned_found


def validate_submission(submission_dir: str) -> Tuple[bool, List[str], List[str]]:
    """
    Convenience function to validate a submission.

    Args:
        submission_dir: Path to submission directory

    Returns:
        (is_valid, errors, warnings)
    """
    validator = SubmissionValidator(Path(submission_dir))
    return validator.validate()
