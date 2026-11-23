"""Build the next FosFolio patch for both installed desktop platforms or one named target."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PLATFORMS = ("windows", "macos")


def build(series_dir, notes, platform=None):
    targets = (platform,) if platform else PLATFORMS
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for target in targets:
            directory = series_dir / target
            previous = json.loads((directory / "latest.json").read_bytes())
            major, minor, patch = map(int, previous["version"].split("."))
            if previous.get("platform") != target or series_dir.name != f"v{major}.{minor}":
                raise ValueError(f"更新清单的平台或系列错误：{directory}")
            version = f"{major}.{minor}.{patch + 1}"
            dest = staging / target
            dest.mkdir()
            shutil.copy2(directory / "更新公告.md", dest)
            subprocess.run([sys.executable, str(Path(__file__).with_name("make_patch.py")), str(dest),
                            "--platform", target, "--version", version, "--min-base", previous["min_base"],
                            "--full-installer", previous["full_installer"], "--in-app-only", "--notes", str(notes)],
                           check=True)
        for target in targets:
            shutil.copytree(staging / target, series_dir / target, dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series_dir", type=Path, help="更新分支的 updates/v0.9 目录")
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--platform", choices=PLATFORMS, help="默认同时构建两个平台")
    args = parser.parse_args()
    build(args.series_dir, args.notes, args.platform)
