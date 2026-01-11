"""
Immutable audit logging for FLAIR.

This module provides an append-only audit trail of all operations
performed during FLAIR benchmark execution. The log is designed
to be tamper-evident and reviewable by site coordinators.

All operations are logged with timestamps, user info, and outcomes.
"""

import json
import logging
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Represents a single audit log entry."""

    timestamp: str
    operation: str
    status: str  # "success", "failure", "blocked"
    details: Dict[str, Any]
    session_id: str
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute hash of this entry for chain integrity."""
        data = {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "status": self.status,
            "details": self.details,
            "session_id": self.session_id,
            "previous_hash": self.previous_hash,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class AuditLog:
    """
    Immutable audit log for FLAIR operations.

    Provides append-only logging with hash chaining for tamper detection.
    Each entry includes a hash of the previous entry, creating a verifiable
    chain of custody.

    Usage:
        log = AuditLog("flair_audit.log")
        log.log_operation("train_model", "success", {"model": "xgboost"})
        log.log_operation("export_results", "blocked", {"reason": "PHI detected"})
    """

    def __init__(self, log_path: str, session_id: Optional[str] = None):
        """
        Initialize audit log.

        Args:
            log_path: Path to the audit log file
            session_id: Unique identifier for this session
        """
        self.log_path = Path(log_path)
        self.session_id = session_id or self._generate_session_id()
        self._last_hash: Optional[str] = None

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Load last hash from existing log if present
        if self.log_path.exists():
            self._load_last_hash()

        # Log session start
        self.log_operation(
            "session_start",
            "success",
            {
                "session_id": self.session_id,
                "flair_version": "1.0.0",
            },
        )

    def _generate_session_id(self) -> str:
        """Generate a unique session identifier."""
        import uuid

        return f"flair_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _load_last_hash(self) -> None:
        """Load the last entry hash from the log file."""
        try:
            with open(self.log_path, "r") as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    self._last_hash = last_entry.get("entry_hash")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load last hash from audit log: {e}")

    def log_operation(
        self,
        operation: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log an operation to the audit trail.

        Args:
            operation: Name of the operation (e.g., "train_model", "export_results")
            status: Status of the operation ("success", "failure", "blocked")
            details: Additional details about the operation

        Returns:
            The created AuditEntry
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            status=status,
            details=details or {},
            session_id=self.session_id,
            previous_hash=self._last_hash,
        )

        # Compute hash including previous hash for chain integrity
        entry.entry_hash = entry.compute_hash()
        self._last_hash = entry.entry_hash

        # Append to log file (append-only)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        logger.debug(f"Audit: {operation} - {status}")

        return entry

    def verify_chain(self) -> bool:
        """
        Verify the integrity of the audit log chain.

        Returns:
            True if chain is valid, False if tampering detected
        """
        if not self.log_path.exists():
            return True

        try:
            with open(self.log_path, "r") as f:
                lines = f.readlines()

            previous_hash = None

            for i, line in enumerate(lines):
                entry_dict = json.loads(line)

                # Verify previous hash matches
                if entry_dict.get("previous_hash") != previous_hash:
                    logger.error(f"Chain broken at entry {i}: previous_hash mismatch")
                    return False

                # Recompute hash and verify
                entry = AuditEntry(
                    timestamp=entry_dict["timestamp"],
                    operation=entry_dict["operation"],
                    status=entry_dict["status"],
                    details=entry_dict["details"],
                    session_id=entry_dict["session_id"],
                    previous_hash=entry_dict.get("previous_hash"),
                )

                computed_hash = entry.compute_hash()
                if computed_hash != entry_dict.get("entry_hash"):
                    logger.error(f"Chain broken at entry {i}: entry_hash mismatch")
                    return False

                previous_hash = entry_dict.get("entry_hash")

            logger.info("Audit log chain verified successfully")
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Audit log verification failed: {e}")
            return False

    def get_entries(
        self,
        operation: Optional[str] = None,
        status: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[AuditEntry]:
        """
        Retrieve entries from the audit log with optional filtering.

        Args:
            operation: Filter by operation name
            status: Filter by status
            session_id: Filter by session ID

        Returns:
            List of matching AuditEntry objects
        """
        if not self.log_path.exists():
            return []

        entries = []

        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    entry_dict = json.loads(line)

                    # Apply filters
                    if operation and entry_dict.get("operation") != operation:
                        continue
                    if status and entry_dict.get("status") != status:
                        continue
                    if session_id and entry_dict.get("session_id") != session_id:
                        continue

                    entries.append(
                        AuditEntry(
                            timestamp=entry_dict["timestamp"],
                            operation=entry_dict["operation"],
                            status=entry_dict["status"],
                            details=entry_dict["details"],
                            session_id=entry_dict["session_id"],
                            previous_hash=entry_dict.get("previous_hash"),
                            entry_hash=entry_dict.get("entry_hash"),
                        )
                    )
                except json.JSONDecodeError:
                    continue

        return entries

    def get_blocked_operations(self) -> List[AuditEntry]:
        """Get all blocked operations (potential violations)."""
        return self.get_entries(status="blocked")


# Global audit log instance
_audit_log: Optional[AuditLog] = None


def init_audit_log(log_path: str) -> AuditLog:
    """Initialize the global audit log."""
    global _audit_log
    _audit_log = AuditLog(log_path)
    return _audit_log


def get_audit_log() -> Optional[AuditLog]:
    """Get the global audit log instance."""
    return _audit_log


def log_operation(
    operation: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
) -> Optional[AuditEntry]:
    """
    Log an operation to the global audit log.

    Args:
        operation: Name of the operation
        status: Status ("success", "failure", "blocked")
        details: Additional details

    Returns:
        AuditEntry if log is initialized, None otherwise
    """
    if _audit_log is None:
        logger.warning(f"Audit log not initialized, operation not logged: {operation}")
        return None

    return _audit_log.log_operation(operation, status, details)


@contextmanager
def audit_operation(operation: str, details: Optional[Dict[str, Any]] = None):
    """
    Context manager for auditing an operation.

    Logs success on normal exit, failure on exception.

    Usage:
        with audit_operation("train_model", {"model": "xgboost"}):
            train_model()
    """
    try:
        yield
        log_operation(operation, "success", details)
    except Exception as e:
        log_operation(
            operation,
            "failure",
            {**(details or {}), "error": str(e), "error_type": type(e).__name__},
        )
        raise
