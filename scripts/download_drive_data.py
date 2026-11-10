"""Download the project's shared dataset while preserving its folder layout."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1] / "data"
FOLDER_ID = "1lusdJKxeES-gXswoCeCXozIY3Xw4guI0"
INVENTORY = ROOT / "drive-file-list.json"


def download_one(item):
    destination = (ROOT / item["path"]).resolve()
    if not destination.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path outside data directory: {item['path']}")
    if destination.is_file() and destination.stat().st_size > 0:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            gdown.download(
                id=item["id"], output=str(destination), quiet=True,
                use_cookies=False, resume=True, timeout=60, retries=2,
            )
            return None
        except Exception as error:
            if attempt == 2:
                return {**item, "error": str(error)}
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    args = parser.parse_args()
    inventory = args.inventory
    if not inventory.exists():
        files = gdown.download_folder(
            id=FOLDER_ID, output=str(ROOT), skip_download=True,
            quiet=True, use_cookies=False, timeout=60,
        )
        inventory.write_text(json.dumps(
            [{"id": f.id, "path": f.path} for f in files],
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")
    items = json.loads(inventory.read_text(encoding="utf-8"))
    items.sort(key=lambda item: ("/images/" in item["path"], "/labels/" in item["path"]))
    print(f"Inventory: {len(items)} files", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(download_one, item) for item in items]
        for count, future in enumerate(as_completed(futures), 1):
            error = future.result()
            if error:
                failed.append(error)
            if count % 100 == 0 or count == len(items):
                print(f"Completed {count}/{len(items)}; failures {len(failed)}", flush=True)
    report = {"total": len(items), "failed": failed}
    (ROOT / "drive-download-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if failed:
        raise SystemExit(f"{len(failed)} downloads failed; see data/drive-download-report.json")


if __name__ == "__main__":
    main()
