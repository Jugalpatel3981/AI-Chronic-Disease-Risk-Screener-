"""Download the NHANES data files used by this project from the CDC website.

Usage (from the repository root):
    python -m ml.nhanes.download
"""

import sys
import time
import urllib.request

from .config import BASE_URL, BP_FILE, CYCLES, FILES, RAW_DIR

USER_AGENT = "nhanes-research-downloader/1.0"
XPT_HEADER = b"HEADER RECORD*******LIBRARY HEADER RECORD"


def files_for_cycle(suffix: str) -> list[str]:
    return FILES + [BP_FILE[suffix]]


def download_file(url: str, dest, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if not data.startswith(XPT_HEADER):
                raise ValueError("response is not a SAS XPT file")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except Exception as exc:  # network errors, bad responses
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            time.sleep(2 * attempt)


def main() -> int:
    failures = []
    for label, (year, suffix) in CYCLES.items():
        cycle_dir = RAW_DIR / label
        cycle_dir.mkdir(parents=True, exist_ok=True)
        for name in files_for_cycle(suffix):
            dest = cycle_dir / f"{name}_{suffix}.xpt"
            if dest.exists():
                print(f"[skip] {label} {dest.name} (already downloaded)")
                continue
            url = BASE_URL.format(year=year, name=name, suffix=suffix)
            try:
                download_file(url, dest)
                print(f"[ok]   {label} {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            except RuntimeError as exc:
                print(f"[FAIL] {exc}")
                failures.append(url)
    if failures:
        print(f"\n{len(failures)} file(s) failed to download.")
        return 1
    print(f"\nAll files saved under {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
