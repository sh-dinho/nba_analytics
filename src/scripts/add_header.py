# ============================================================
# File: scripts/add_headers.py
# Purpose: Enforce standardized headers across Python files (v2.0)
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

import sys
from pathlib import Path

HEADER_TEMPLATE = """# ============================================================
# File: {file_path}
# Purpose: {purpose}
# Version: {version}
# Author: {author}
# Date: {date}
# ============================================================

"""


def add_header_to_file(path: Path, purpose: str, version: str, author: str, date: str):
    text = path.read_text(encoding="utf-8")
    if text.strip().startswith(
        "# ============================================================"
    ):
        return
    header = HEADER_TEMPLATE.format(
        file_path=path.as_posix(),
        purpose=purpose,
        version=version,
        author=author,
        date=date,
    )
    path.write_text(header + text, encoding="utf-8")


def main():
    root = Path(sys.argv[1])
    purpose = sys.argv[2] if len(sys.argv) > 2 else "NBA Analytics Pipeline"
    version = sys.argv[3] if len(sys.argv) > 3 else "2.0"
    author = sys.argv[4] if len(sys.argv) > 4 else "Your Team"
    date = sys.argv[5] if len(sys.argv) > 5 else "December 2025"
    for py in root.rglob("*.py"):
        add_header_to_file(py, purpose, version, author, date)


if __name__ == "__main__":
    main()
