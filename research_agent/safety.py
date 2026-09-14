from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_BLOCKED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "metadata.google.internal",
    "metadata.google.internal.",
}

_BLOCKED_SUFFIXES = (".local", ".localhost", ".internal")


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def host_is_blocked(host: str, *, resolve: bool = True) -> bool:
    hostname = (host or "").strip().lower().rstrip(".")
    if not hostname or hostname in _BLOCKED_HOSTS:
        return True
    if any(hostname.endswith(suffix) for suffix in _BLOCKED_SUFFIXES):
        return True
    try:
        ip = ipaddress.ip_address(hostname)
        return _ip_is_blocked(ip)
    except ValueError:
        pass
    if not resolve:
        return False
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True
    for info in infos:
        addr = info[4][0]
        try:
            if _ip_is_blocked(ipaddress.ip_address(addr)):
                return True
        except ValueError:
            continue
    return False


def is_safe_http_url(url: str, *, resolve: bool = True) -> bool:
    """Return True only for public http(s) URLs. Read-only fetch guard."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.username or parsed.password:
        return False
    host = parsed.hostname
    if not host:
        return False
    return not host_is_blocked(host, resolve=resolve)
