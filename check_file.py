

from pathlib import Path
from datetime import datetime
from typing import List
import argparse

def files_by_mtime(folder: str, newest_first: bool = True, include_dirs: bool = False) -> List[Path]:
    """Return paths sorted by last modified time."""
    p = Path("C:\\Users\\conor\\DemoLaiho\\demoparser\\demofiles").expanduser().resolve()
    entries = [e for e in p.iterdir() if e.is_file() or (include_dirs and e.is_dir())]
    entries.sort(key=lambda e: e.stat().st_mtime, reverse=False)
    return entries

def main():
    parser = argparse.ArgumentParser(description="List files sorted by date modified.")
    parser.add_argument("folder", nargs="?", default=".", help="Folder to list")
    parser.add_argument("--oldest-first", action="store_true", help="Sort oldest to newest")
    parser.add_argument("--include-dirs", action="store_true", help="Include directories")
    parser.add_argument("--out", default="file_list.txt", help="Output text file name")
    args = parser.parse_args()

    lines = []
    for path in files_by_mtime(args.folder, newest_first=not args.oldest_first, include_dirs=args.include_dirs):
        st = path.stat()
        ts = datetime.fromtimestamp(st.st_mtime).isoformat(sep=" ", timespec="seconds")
        line = f"{ts}  {st.st_size:>10}  {path.name}"
        print(line)            # Print to console
        lines.append(line)     # Store for writing

    # Save to text file
    output_path = Path(args.out)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved output to {output_path}")

if __name__ == "__main__":
    main()
