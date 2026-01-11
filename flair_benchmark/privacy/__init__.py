"""Privacy enforcement module for FLAIR.

This module provides critical privacy protections for the FLAIR benchmark:
1. Network blocking - Prevents any network requests from submitted code
2. PHI detection - Scans outputs for protected health information patterns
3. Audit logging - Immutable record of all operations

WARNING: Violations of privacy policies will result in submission rejection
and potential ban from the FLAIR benchmark.
"""

from flair_benchmark.privacy.network_guard import NetworkBlocker, network_blocked
from flair_benchmark.privacy.phi_detector import PHIDetector, PHI_PATTERNS
from flair_benchmark.privacy.audit_log import AuditLog, log_operation

__all__ = [
    "NetworkBlocker",
    "network_blocked",
    "PHIDetector",
    "PHI_PATTERNS",
    "AuditLog",
    "log_operation",
]
