"""Fetch Near-Earth Object data from NASA NeoWs (Near Earth Object Web Service).

We use the "browse" endpoint to retrieve near-earth objects page by page, then cache
the raw JSON responses to disk (so the pipeline is reproducible and doesn't re-hit the API).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from my_project.config import NASA_NEOWS_BASE_URL, Paths
from my_project.logger import logger


def _http_get_json(url: str, timeout_s: int = 30) -> dict[str, Any]:
    """GET a URL and parse JSON.

    Args:
        url: Full URL to request.
        timeout_s: Request timeout in seconds.

    Returns:
        Parsed JSON as dict.

    Raises:
        RuntimeError: If HTTP status is not 200 or JSON parsing fails.
    """
    logger.debug("GET %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "neo-hazard-lab/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        status = getattr(resp, "status", 200)
        body = resp.read().decode("utf-8")
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {body[:200]}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to parse JSON: {exc}") from exc


def fetch_browse_page(api_key: str, page: int, size: int = 20) -> dict[str, Any]:
    """Fetch one browse page from NeoWs.

    Args:
        api_key: NASA API key (DEMO_KEY works for small demos).
        page: Page index (0-based).
        size: Page size (NeoWs supports size parameter).

    Returns:
        JSON response dict.
    """
    params = {"api_key": api_key, "page": str(page), "size": str(size)}
    url = f"{NASA_NEOWS_BASE_URL}/neo/browse?{urllib.parse.urlencode(params)}"
    return _http_get_json(url)


def cache_page(payload: dict[str, Any], out_path: Path) -> None:
    """Write JSON payload to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Cached %s", out_path)


def fetch_and_cache_pages(api_key: str, pages: int, size: int, sleep_s: float = 0.2) -> list[Path]:
    """Fetch multiple pages and cache them.

    Args:
        api_key: NASA API key.
        pages: Number of pages to fetch.
        size: Objects per page.
        sleep_s: Small sleep between requests to be polite.

    Returns:
        List of cached file paths.
    """
    paths = Paths.from_here()
    paths.data_raw.mkdir(parents=True, exist_ok=True)

    cached: list[Path] = []
    for p in range(pages):
        payload = fetch_browse_page(api_key=api_key, page=p, size=size)
        out_path = paths.data_raw / f"neows_browse_page_{p:03d}.json"
        cache_page(payload, out_path)
        cached.append(out_path)
        time.sleep(sleep_s)
    return cached


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Fetch NASA NeoWs browse pages and cache raw JSON.")
    parser.add_argument("--api-key", default="DEMO_KEY", help="NASA API key (default: DEMO_KEY).")
    parser.add_argument("--pages", type=int, default=3, help="Number of pages to fetch (default: 3).")
    parser.add_argument("--size", type=int, default=30, help="Objects per page (default: 30).")
    args = parser.parse_args()

    logger.info("Fetching NeoWs browse pages: pages=%s size=%s", args.pages, args.size)
    files = fetch_and_cache_pages(api_key=args.api_key, pages=args.pages, size=args.size)
    logger.info("Done. Cached %d files.", len(files))


if __name__ == "__main__":
    main()
