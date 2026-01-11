"""
PHI (Protected Health Information) detection for FLAIR.

This module scans outputs for potential PHI patterns before allowing
data to leave the federated site. Detects common identifiers, dates,
and other patterns that could identify individual patients.

All outputs are scanned before release. Detection triggers review.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# PHI pattern definitions
# These patterns are designed to catch common PHI formats
PHI_PATTERNS: Dict[str, str] = {
    # Direct identifiers
    "hospitalization_id": r"hospitalization_id\s*[=:]\s*[A-Za-z0-9_-]+",
    "patient_id": r"patient_id\s*[=:]\s*[A-Za-z0-9_-]+",
    "mrn": r"\bMRN\s*[:#]?\s*\d{5,}",
    "ssn": r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",
    # Names (basic pattern - catches "Patient: John Smith" etc.)
    "name_pattern": r"\b(patient|name|doctor|physician)\s*[:#]?\s*[A-Z][a-z]+\s+[A-Z][a-z]+",
    # Contact info
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    # Dates that could be PHI (birth dates, specific admission dates)
    "date_of_birth": r"\b(DOB|date.?of.?birth|birth.?date)\s*[:#]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
    "specific_date": r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO format with time
    # Addresses
    "address": r"\b\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Court|Ct)\b",
    "zip_code": r"\b\d{5}(-\d{4})?\b",
    # Medical record numbers in various formats
    "record_number": r"\b(record|chart|case)\s*[:#]?\s*\d{6,}",
    # Account/billing numbers
    "account_number": r"\b(account|acct|billing)\s*[:#]?\s*\d{6,}",
}


@dataclass
class PHIViolation:
    """Represents a detected PHI violation."""

    pattern_name: str
    count: int
    samples: List[str]
    severity: str  # "high", "medium", "low"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern_name": self.pattern_name,
            "count": self.count,
            "samples": self.samples[:3],  # Only include first 3 samples
            "severity": self.severity,
        }


class PHIDetector:
    """
    Detect potential PHI in text outputs.

    This class scans text for patterns that might indicate protected
    health information. All outputs must pass PHI detection before
    being released from the federated site.

    Usage:
        detector = PHIDetector()
        violations = detector.scan(output_text)
        if violations:
            raise PHIViolationError(violations)
    """

    # Pattern severity levels
    SEVERITY_MAP = {
        "hospitalization_id": "high",
        "patient_id": "high",
        "mrn": "high",
        "ssn": "high",
        "name_pattern": "high",
        "phone": "medium",
        "email": "medium",
        "date_of_birth": "high",
        "specific_date": "low",
        "address": "high",
        "zip_code": "low",
        "record_number": "high",
        "account_number": "medium",
    }

    def __init__(self, patterns: Optional[Dict[str, str]] = None):
        """
        Initialize PHI detector.

        Args:
            patterns: Custom PHI patterns (defaults to PHI_PATTERNS)
        """
        self.patterns = patterns or PHI_PATTERNS
        self.compiled = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }

    def scan(self, text: str) -> List[PHIViolation]:
        """
        Scan text for PHI patterns.

        Args:
            text: Text to scan

        Returns:
            List of PHIViolation objects for each detected pattern
        """
        if not text:
            return []

        violations = []

        for name, pattern in self.compiled.items():
            matches = pattern.findall(text)
            if matches:
                # Truncate samples to avoid including too much PHI in logs
                samples = [m[:50] + "..." if len(m) > 50 else m for m in matches[:5]]
                severity = self.SEVERITY_MAP.get(name, "medium")

                violations.append(
                    PHIViolation(
                        pattern_name=name,
                        count=len(matches),
                        samples=samples,
                        severity=severity,
                    )
                )

        if violations:
            logger.warning(
                f"PHI detection found {len(violations)} pattern types in output"
            )

        return violations

    def has_phi(self, text: str) -> bool:
        """
        Quick check if text contains any PHI.

        Args:
            text: Text to check

        Returns:
            True if PHI detected, False otherwise
        """
        return len(self.scan(text)) > 0

    def has_high_severity_phi(self, text: str) -> bool:
        """
        Check if text contains high-severity PHI.

        Args:
            text: Text to check

        Returns:
            True if high-severity PHI detected
        """
        violations = self.scan(text)
        return any(v.severity == "high" for v in violations)

    def scan_dataframe(self, df: Any) -> List[PHIViolation]:
        """
        Scan a DataFrame for PHI.

        Checks column names and string column values.

        Args:
            df: pandas or polars DataFrame

        Returns:
            List of PHIViolation objects
        """
        violations = []

        # Check column names
        for col in df.columns:
            col_violations = self.scan(str(col))
            violations.extend(col_violations)

        # Check string column values (sample first 100 rows)
        try:
            # Handle both pandas and polars
            if hasattr(df, "to_pandas"):
                # Polars DataFrame
                sample_df = df.head(100).to_pandas()
            else:
                # Pandas DataFrame
                sample_df = df.head(100)

            for col in sample_df.columns:
                if sample_df[col].dtype == "object":
                    for val in sample_df[col].dropna().astype(str):
                        col_violations = self.scan(str(val))
                        violations.extend(col_violations)
        except Exception as e:
            logger.warning(f"Could not fully scan DataFrame: {e}")

        return violations

    def scan_dict(self, data: Dict[str, Any]) -> List[PHIViolation]:
        """
        Recursively scan a dictionary for PHI.

        Args:
            data: Dictionary to scan

        Returns:
            List of PHIViolation objects
        """
        violations = []

        def _scan_value(value: Any) -> None:
            if isinstance(value, str):
                violations.extend(self.scan(value))
            elif isinstance(value, dict):
                for k, v in value.items():
                    violations.extend(self.scan(str(k)))
                    _scan_value(v)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    _scan_value(item)

        for key, value in data.items():
            violations.extend(self.scan(str(key)))
            _scan_value(value)

        return violations


class PHIViolationError(Exception):
    """Exception raised when PHI is detected in output."""

    def __init__(self, violations: List[PHIViolation]):
        self.violations = violations
        violation_summary = ", ".join(
            f"{v.pattern_name} ({v.count} instances)" for v in violations
        )
        super().__init__(
            f"PHI detected in output: {violation_summary}. "
            "Output cannot be released without review."
        )


def validate_output_no_phi(text: str, allow_low_severity: bool = False) -> bool:
    """
    Validate that output contains no PHI.

    Args:
        text: Text to validate
        allow_low_severity: If True, only block high/medium severity

    Returns:
        True if validation passes

    Raises:
        PHIViolationError: If PHI is detected
    """
    detector = PHIDetector()
    violations = detector.scan(text)

    if allow_low_severity:
        violations = [v for v in violations if v.severity != "low"]

    if violations:
        raise PHIViolationError(violations)

    return True
