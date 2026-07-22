"""Camofox browser wrapper — local anti-detection browser server (Docker, :9377).

C++ engine-level fingerprint spoofing (Camoufox Firefox fork). Used when
needs_rendering=True and the task requires JS-heavy pages or stealth.

API flow (verified against v2.4.6):
  1. POST /start {userId}              -> creates session + profile
  2. POST /tabs {userId, sessionKey, url} -> opens tab, returns targetId
  3. GET  /snapshot?userId&sessionKey&targetId -> accessibility tree (text, LLM-friendly)
  4. POST /act {userId, sessionKey, targetId, kind, ref} -> click/type/etc
"""
from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import settings


TIMEOUT = 60.0


class CamofoxError(Exception):
    pass


def _headers() -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    # Camofox requires the bearer token when CAMOFOX_AUTH_MODE=auto and host is non-loopback.
    if settings.camofox_api_key:
        h["Authorization"] = f"Bearer {settings.camofox_api_key}"
    return h


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
)
def health() -> bool:
    """Check if Camofox server is up and the browser engine is connected."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{settings.camofox_url}/health", headers=_headers())
        data = resp.json()
        return resp.status_code == 200 and data.get("ok") is True
    except Exception:
        return False


def _start_session(session_id: str) -> None:
    """Create a session + profile for the given userId."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            f"{settings.camofox_url}/start",
            json={"userId": session_id},
            headers=_headers(),
        )
    if resp.status_code >= 400:
        raise CamofoxError(f"Camofox start HTTP {resp.status_code}: {resp.text[:200]}")


def navigate(url: str, session_id: str | None = None) -> dict:
    """Open a URL in a new tab. Returns {targetId, url, snapshot}."""
    sid = session_id or "orchestrator"
    session_key = "default"
    # Ensure session exists
    try:
        _start_session(sid)
    except CamofoxError:
        pass  # session may already exist
    # Open tab
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            f"{settings.camofox_url}/tabs",
            json={"userId": sid, "sessionKey": session_key, "url": url},
            headers=_headers(),
        )
    if resp.status_code >= 400:
        raise CamofoxError(f"Camofox tabs HTTP {resp.status_code}: {resp.text[:200]}")
    tab = resp.json()
    target_id = tab.get("targetId") or tab.get("tabId")
    if not target_id:
        raise CamofoxError(f"Camofox no targetId in response: {tab}")
    # Give the browser a moment to render
    import time
    time.sleep(1.5)
    # Fetch snapshot immediately
    snap = _snapshot(sid, session_key, target_id)
    return {"targetId": target_id, "url": url, "snapshot": snap, "userId": sid, "sessionKey": session_key}


def _snapshot(session_id: str, session_key: str, target_id: str) -> str:
    """Get an accessibility-tree snapshot (text-based, LLM-friendly)."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(
            f"{settings.camofox_url}/snapshot",
            params={"userId": session_id, "sessionKey": session_key, "targetId": target_id},
            headers=_headers(),
        )
    if resp.status_code >= 400:
        raise CamofoxError(f"Camofox snapshot HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    return data.get("snapshot") or data.get("content") or str(data)[:5000]


def snapshot(session_id: str, target_id: str | None = None, session_key: str = "default") -> str:
    """Get snapshot. If target_id not given, fetches the first tab."""
    if target_id:
        return _snapshot(session_id, session_key, target_id)
    # List tabs and snapshot the first
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(
            f"{settings.camofox_url}/tabs",
            params={"userId": session_id},
            headers=_headers(),
        )
    tabs = resp.json().get("tabs", [])
    if not tabs:
        raise CamofoxError("No tabs open for session")
    target_id = tabs[0].get("targetId") or tabs[0].get("tabId")
    return _snapshot(session_id, session_key, target_id)


def click(session_id: str, ref: str, target_id: str, session_key: str = "default") -> dict:
    """Click an element by its eN ref (from snapshot)."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            f"{settings.camofox_url}/act",
            json={"userId": session_id, "sessionKey": session_key, "targetId": target_id,
                  "kind": "click", "ref": ref},
            headers=_headers(),
        )
    if resp.status_code >= 400:
        raise CamofoxError(f"Camofox click HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def screenshot(session_id: str, target_id: str, session_key: str = "default") -> str:
    """Take a screenshot."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(
            f"{settings.camofox_url}/screenshot",
            params={"userId": session_id, "sessionKey": session_key, "targetId": target_id},
            headers=_headers(),
        )
    if resp.status_code >= 400:
        raise CamofoxError(f"Camofox screenshot HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    return data.get("screenshot") or data.get("url") or str(data)[:200]
