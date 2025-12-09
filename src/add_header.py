# ============================================================
# Path: scripts/add_header.py
# File: add_header.py
# Purpose: Pre-commit hook script to enforce standardized headers in Python files
# Project: nba_analysis
# ============================================================

import pathlib
import sys

HEADER_TEMPLATE = """# ============================================================
# Path: {path}
# File: {file}
# Purpose: <add short description here>
# Project: nba_analysis
# ============================================================

"""

EXPECTED_START = "# ============================================================"
EXPECTED_PROJECT = "# Project: nba_analysis"


def has_valid_header(content: list[str]) -> bool:
    """Check if the file already has a valid standardized header."""
    return (
        len(content) >= 5
        and content[0].startswith(EXPECTED_START)
        and any(EXPECTED_PROJECT in line for line in content[:6])
    )


def ensure_header(file_path: pathlib.Path):
    """Ensure the file has a standardized header, fixing malformed ones if needed."""
    content = file_path.read_text(encoding="utf-8").splitlines()

    if has_valid_header(content):
        # Already valid, do nothing
        return

    # Always replace with a fresh header
    header = HEADER_TEMPLATE.format(path=file_path.as_posix(), file=file_path.name)
    new_content = header + "\n".join(content)
    file_path.write_text(new_content, encoding="utf-8")


def main():
    for filename in sys.argv[1:]:
        path = pathlib.Path(filename)
        if path.suffix == ".py":
            ensure_header(path)


if __name__ == "__main__":
    main()
