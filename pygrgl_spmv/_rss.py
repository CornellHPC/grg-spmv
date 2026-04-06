"""Small RSS checkpoint helper shared by compile and backend setup logs."""

from __future__ import annotations

import logging
import os


def rss_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        with open("/proc/self/statm", encoding="ascii") as handle:
            fields = handle.readline().split()
    except (OSError, ValueError):
        return None
    if len(fields) < 2:
        return None
    try:
        return int(fields[1]) * page_size
    except ValueError:
        return None


def rss_checkpoint(
    logger: logging.Logger,
    label: str,
    prev_rss_bytes: int | None,
    *,
    extras: tuple[tuple[str, object], ...] = (),
) -> int | None:
    if not logger.isEnabledFor(logging.INFO):
        return prev_rss_bytes
    value = rss_bytes()
    if value is None:
        return prev_rss_bytes
    suffix = ""
    if extras:
        suffix = " " + " ".join(f"{key}={item}" for key, item in extras)
    if prev_rss_bytes is None:
        logger.info(
            "rss %s rss_bytes=%d rss_mib=%.1f%s",
            label,
            value,
            value / (1024.0 * 1024.0),
            suffix,
        )
    else:
        delta = int(value - prev_rss_bytes)
        logger.info(
            "rss %s rss_bytes=%d rss_mib=%.1f delta_bytes=%+d delta_mib=%+.1f%s",
            label,
            value,
            value / (1024.0 * 1024.0),
            delta,
            delta / (1024.0 * 1024.0),
            suffix,
        )
    return value


__all__ = ["rss_bytes", "rss_checkpoint"]
