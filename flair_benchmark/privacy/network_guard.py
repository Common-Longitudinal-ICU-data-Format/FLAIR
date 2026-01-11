"""
Network blocking implementation for FLAIR.

This module blocks ALL network access from submitted methods by monkey-patching
the Python socket module. This is a critical privacy protection to prevent
data exfiltration from federated sites.

WARNING: Any attempt to circumvent network blocking will result in immediate
submission rejection and potential ban from the FLAIR benchmark.
"""

import socket
import functools
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


class NetworkBlocker:
    """
    Context manager and utility class that blocks all network access.

    Uses socket monkey-patching to prevent ANY network requests from
    submitted code. This is the primary defense against data exfiltration.

    Usage:
        # Block network for duration of session
        NetworkBlocker.block()

        # Or use as context manager
        with NetworkBlocker.context():
            # Network blocked here
            run_submitted_code()

        # Or use decorator
        @network_blocked
        def my_function():
            pass
    """

    _original_socket = None
    _original_getaddrinfo = None
    _blocked = False
    _block_count = 0

    @classmethod
    def block(cls) -> None:
        """
        Block all network sockets.

        Once called, any attempt to create a socket will raise PermissionError.
        This includes HTTP requests, DNS lookups, and any other network access.
        """
        if cls._blocked:
            cls._block_count += 1
            return

        # Save original socket
        cls._original_socket = socket.socket
        cls._original_getaddrinfo = socket.getaddrinfo

        def blocked_socket(*args: Any, **kwargs: Any) -> None:
            raise PermissionError(
                "\n"
                "=" * 70 + "\n"
                "FLAIR PRIVACY VIOLATION: Network access is BLOCKED\n"
                "=" * 70 + "\n\n"
                "Submitted methods cannot make any network requests.\n"
                "This includes:\n"
                "  - HTTP/HTTPS requests (requests, urllib, httpx, etc.)\n"
                "  - Socket connections (socket, websocket, etc.)\n"
                "  - DNS lookups\n"
                "  - FTP, SMTP, SSH, and all other protocols\n\n"
                "If this violation is found during review, your submission\n"
                "will be REJECTED and you may be BANNED from FLAIR.\n\n"
                "PIs are doing you a favor by running your code on their data.\n"
                "Respect their trust and protect patient privacy.\n"
                "=" * 70
            )

        def blocked_getaddrinfo(*args: Any, **kwargs: Any) -> None:
            raise PermissionError(
                "FLAIR PRIVACY VIOLATION: DNS lookups are blocked. "
                "Network access is not permitted in submitted methods."
            )

        socket.socket = blocked_socket
        socket.getaddrinfo = blocked_getaddrinfo
        cls._blocked = True
        cls._block_count = 1

        logger.info("Network access blocked for FLAIR privacy protection")

    @classmethod
    def unblock(cls) -> None:
        """
        Restore network access.

        WARNING: This should only be called by FLAIR internal code,
        never by submitted methods. Attempting to call this from
        submitted code will be detected during review.
        """
        if not cls._blocked:
            return

        cls._block_count -= 1
        if cls._block_count > 0:
            return

        if cls._original_socket is not None:
            socket.socket = cls._original_socket
            cls._original_socket = None

        if cls._original_getaddrinfo is not None:
            socket.getaddrinfo = cls._original_getaddrinfo
            cls._original_getaddrinfo = None

        cls._blocked = False
        logger.info("Network access restored (FLAIR internal)")

    @classmethod
    def is_blocked(cls) -> bool:
        """Check if network is currently blocked."""
        return cls._blocked

    @classmethod
    def context(cls) -> "NetworkBlockerContext":
        """Return a context manager for network blocking."""
        return NetworkBlockerContext()


class NetworkBlockerContext:
    """Context manager for network blocking."""

    def __enter__(self) -> "NetworkBlockerContext":
        NetworkBlocker.block()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Note: We intentionally do NOT unblock on exit for submitted code
        # The network should stay blocked for the entire session
        pass


def network_blocked(func: Callable) -> Callable:
    """
    Decorator to run a function with network blocked.

    Usage:
        @network_blocked
        def run_user_model(data):
            # Network is blocked here
            model.predict(data)
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        NetworkBlocker.block()
        try:
            return func(*args, **kwargs)
        finally:
            # Note: We do NOT unblock - network stays blocked
            pass

    return wrapper


def verify_network_blocked() -> bool:
    """
    Verify that network blocking is active.

    Returns True if blocking is working, raises exception otherwise.
    """
    if not NetworkBlocker.is_blocked():
        raise RuntimeError(
            "FLAIR SECURITY ERROR: Network blocking is not active. "
            "Cannot run submitted code without network protection."
        )

    # Try to actually create a socket to verify blocking works
    try:
        import socket as test_socket

        test_socket.socket(test_socket.AF_INET, test_socket.SOCK_STREAM)
        raise RuntimeError(
            "FLAIR SECURITY ERROR: Network blocking failed verification. "
            "Socket creation should have been blocked."
        )
    except PermissionError:
        # Expected - blocking is working
        return True


# List of banned packages that should never be imported in submitted code
BANNED_PACKAGES = frozenset(
    {
        # HTTP libraries
        "requests",
        "urllib3",
        "httpx",
        "aiohttp",
        "httplib2",
        "pycurl",
        # Low-level network
        "socket",
        "ssl",
        "websocket",
        "websockets",
        # Remote access
        "paramiko",
        "fabric",
        "ftplib",
        "smtplib",
        "imaplib",
        "poplib",
        "telnetlib",
        "nntplib",
        # Cloud SDKs (could exfiltrate data)
        "boto3",
        "botocore",
        "google.cloud",
        "azure",
        # Database clients with network access
        "pymongo",
        "redis",
        "elasticsearch",
    }
)
