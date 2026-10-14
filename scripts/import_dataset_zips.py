"""Import downloaded Drive ZIP parts, preserving any conflicting local files."""

import json
import shutil
import stat
import zipfile
import zlib
from datetime import datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1] / "data"
DOWNLOADS = Path.home() / "Downloads"
ALLOWED = {"raw", "processed", "audit", "tools"}


def crc32(path):
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def main():
    archives = sorted(p for p in DOWNLOADS.glob("*-20261008T*.zip")
                      if p.name.split("-")[0] in ALLOWED)
    if len(archives) != 8:
        raise ValueError(f"Expected 8 ZIP parts, found {len(archives)}")
    root = ROOT.resolve()
    for archive in archives:
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                relative = PurePosixPath(entry.filename)
                if (relative.is_absolute() or not relative.parts or relative.parts[0] not in ALLOWED
                        or ".." in relative.parts or "\\" in entry.filename or ":" in entry.filename):
                    raise ValueError(f"Unexpected ZIP path: {entry.filename}")
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f"ZIP contains a symlink: {entry.filename}")
    backup = root / "import-backup" / datetime.now().strftime("%Y%m%d-%H%M%S")
    report = {"archives": [], "extracted": 0, "unchanged": 0, "backed_up": []}
    for archive in archives:
        print(f"Extracting {archive.name}", flush=True)
        with zipfile.ZipFile(archive) as zipped:
            for index, entry in enumerate(zipped.infolist(), 1):
                target = root / entry.filename
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if target.is_file() and target.stat().st_size == entry.file_size and crc32(target) == entry.CRC:
                    report["unchanged"] += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".importing")
                with zipped.open(entry) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                if target.exists():
                    saved = backup / entry.filename
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    target.replace(saved)
                    report["backed_up"].append(entry.filename)
                temporary.replace(target)
                report["extracted"] += 1
                if index % 5000 == 0:
                    print(f"  {index}/{len(zipped.infolist())} entries", flush=True)
        report["archives"].append({"name": archive.name, "crc_verified": True})
        print(f"Finished {archive.name}", flush=True)
    (root / "zip-import-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
