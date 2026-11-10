"""Select archives and metadata; image trees can be restored from archives."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
state = json.loads((ROOT / "drive-folder-state.json").read_text(encoding="utf-8"))
rows = []


def walk(folder_id, prefix):
    for child_id, name, mime in state.get(folder_id, []):
        path = prefix / name
        if mime == "application/vnd.google-apps.folder":
            walk(child_id, path)
        elif path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".txt"} or "audit" in path.parts:
            rows.append({"id": child_id, "path": str(path)})


walk("1lusdJKxeES-gXswoCeCXozIY3Xw4guI0", Path())
(ROOT / "metadata-drive-files.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
)
print(f"Metadata and archives: {len(rows)}")
for row in rows:
    if row["path"].endswith(".zip"):
        print(row["path"])
