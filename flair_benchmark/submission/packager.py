"""
Results packaging for FLAIR benchmark.

Packages results for submission with PHI checks and standardized format.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib

from flair_benchmark.privacy.phi_detector import PHIDetector, PHIViolationError
from flair_benchmark.privacy.audit_log import log_operation

logger = logging.getLogger(__name__)


class ResultsPackager:
    """
    Package benchmark results for submission.

    Ensures all outputs are PHI-free and in the correct format.
    """

    def __init__(
        self,
        output_dir: str,
        site_name: str,
        task_name: str,
        model_name: str,
    ):
        """
        Initialize packager.

        Args:
            output_dir: Directory to write packaged results
            site_name: Site identifier
            task_name: Task identifier
            model_name: Model/method name
        """
        self.output_dir = Path(output_dir)
        self.site_name = site_name
        self.task_name = task_name
        self.model_name = model_name
        self.phi_detector = PHIDetector()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def package(
        self,
        metrics: Dict[str, Any],
        table1: Dict[str, Any],
        tripod_report: Optional[Dict[str, Any]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Package results for submission.

        Args:
            metrics: Model performance metrics
            table1: Table 1 summary statistics
            tripod_report: TRIPOD-AI report (optional)
            additional_metadata: Extra metadata to include

        Returns:
            Path to the packaged results file
        """
        # Validate no PHI in any output
        self._validate_no_phi(metrics, "metrics")
        self._validate_no_phi(table1, "table1")
        if tripod_report:
            self._validate_no_phi(tripod_report, "tripod_report")

        # Create results package
        package = {
            "flair_version": "1.0.0",
            "submission_timestamp": datetime.now().isoformat(),
            "site": self.site_name,
            "task": self.task_name,
            "model": self.model_name,
            "metrics": metrics,
            "table1": table1,
            "tripod_report": tripod_report,
            "metadata": additional_metadata or {},
        }

        # Compute package hash for integrity verification
        package_json = json.dumps(package, sort_keys=True)
        package["integrity_hash"] = hashlib.sha256(package_json.encode()).hexdigest()

        # Save package
        output_path = self.output_dir / f"results_{self.task_name}_{self.model_name}.json"
        with open(output_path, "w") as f:
            json.dump(package, f, indent=2)

        # Log the packaging operation
        log_operation(
            "package_results",
            "success",
            {
                "output_path": str(output_path),
                "task": self.task_name,
                "model": self.model_name,
            },
        )

        logger.info(f"Packaged results to {output_path}")
        return output_path

    def _validate_no_phi(self, data: Dict[str, Any], name: str) -> None:
        """
        Validate that a data structure contains no PHI.

        Args:
            data: Data to validate
            name: Name for error messages

        Raises:
            PHIViolationError: If PHI is detected
        """
        violations = self.phi_detector.scan_dict(data)

        if violations:
            high_severity = [v for v in violations if v.severity == "high"]
            if high_severity:
                log_operation(
                    "phi_detection",
                    "blocked",
                    {"data": name, "violations": [v.to_dict() for v in high_severity]},
                )
                raise PHIViolationError(high_severity)

    @staticmethod
    def verify_package(package_path: str) -> bool:
        """
        Verify the integrity of a results package.

        Args:
            package_path: Path to the package JSON file

        Returns:
            True if package is valid
        """
        with open(package_path, "r") as f:
            package = json.load(f)

        stored_hash = package.pop("integrity_hash", None)
        if stored_hash is None:
            logger.error("Package missing integrity_hash")
            return False

        # Recompute hash
        package_json = json.dumps(package, sort_keys=True)
        computed_hash = hashlib.sha256(package_json.encode()).hexdigest()

        if computed_hash != stored_hash:
            logger.error("Package integrity check failed - hash mismatch")
            return False

        logger.info("Package integrity verified")
        return True


def create_results_summary(results: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of results.

    Args:
        results: Results dictionary

    Returns:
        Formatted string summary
    """
    lines = [
        "=" * 60,
        "FLAIR Benchmark Results Summary",
        "=" * 60,
        "",
        f"Site: {results.get('site', 'Unknown')}",
        f"Task: {results.get('task', 'Unknown')}",
        f"Model: {results.get('model', 'Unknown')}",
        f"Timestamp: {results.get('submission_timestamp', 'Unknown')}",
        "",
        "-" * 60,
        "Performance Metrics",
        "-" * 60,
    ]

    metrics = results.get("metrics", {})
    for metric, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {metric}: {value:.4f}")
        else:
            lines.append(f"  {metric}: {value}")

    lines.extend(
        [
            "",
            "-" * 60,
            "Cohort Summary",
            "-" * 60,
        ]
    )

    table1 = results.get("table1", {})
    lines.append(f"  Total N: {table1.get('n_total', 'Unknown')}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
