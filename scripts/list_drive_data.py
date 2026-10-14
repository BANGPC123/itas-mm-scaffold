"""Inventory the shared dataset, checkpointing each successfully read folder."""

import importlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1] / "data"
STATE = ROOT / "drive-folder-state.json"
MODULE = importlib.import_module("gdown.download_folder")


def read_folder(folder_id):
    for attempt in range(3):
        try:
            with requests.Session() as session:
                return MODULE._parse_embedded_folder_view(
                    sess=session, folder_id=folder_id, verify=True, timeout=90,
                )[1]
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    files = []
    seen = set()
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = {}

        def submit(folder_id, prefix):
            if folder_id in seen:
                return
            seen.add(folder_id)
            if folder_id in state:
                process(folder_id, prefix, state[folder_id])
            else:
                pending[pool.submit(read_folder, folder_id)] = (folder_id, prefix)

        def process(folder_id, prefix, children):
            for child_id, name, mime in children:
                path = str(Path(prefix) / name)
                if mime == "application/vnd.google-apps.folder":
                    submit(child_id, path)
                else:
                    files.append({"id": child_id, "path": path})

        submit("1lusdJKxeES-gXswoCeCXozIY3Xw4guI0", "")
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                folder_id, prefix = pending.pop(future)
                children = future.result()
                state[folder_id] = children
                for attempt in range(5):
                    try:
                        STATE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                        break
                    except OSError:
                        if attempt == 4:
                            raise
                        time.sleep(1)
                process(folder_id, prefix, children)
                print(f"Scanned {len(state)} folders; {len(files)} files; pending {len(pending)}", flush=True)
    (ROOT / "drive-file-list.json").write_text(
        json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Inventory complete: {len(files)} files", flush=True)


if __name__ == "__main__":
    main()
