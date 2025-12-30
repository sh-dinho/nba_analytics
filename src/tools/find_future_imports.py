from __future__ import annotations
import os
import re

from debugpy.server.cli import TARGET

ROOT = os.path.dirname(os.path.dirname(__file__))


def fix_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove ALL occurrences of the future import
    cleaned = [line for line in lines if TARGET not in line]

    # Determine where to insert the import
    insert_at = 0

    # Skip shebang
    if cleaned and cleaned[0].startswith("#!"):
        insert_at = 1

    # Skip encoding declaration
    if len(cleaned) > insert_at and "coding" in cleaned[insert_at]:
        insert_at += 1

    # Skip module docstring
    if len(cleaned) > insert_at and cleaned[insert_at].lstrip().startswith(
        ('"""', "'''")
    ):
        quote = cleaned[insert_at].strip()[:3]
        insert_at += 1
        # Skip until closing docstring
        while insert_at < len(cleaned) and quote not in cleaned[insert_at]:
            insert_at += 1
        insert_at += 1  # skip closing line

    # Insert the import
    cleaned.insert(insert_at, TARGET + "\n")

    # Write back
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(cleaned)

    print(f"✔ Fixed: {path}")


def scan_and_fix():
    for root, dirs, files in os.walk(ROOT):
        for file in files:
            if not file.endswith(".py"):
                continue

            path = os.path.join(root, file)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # If file contains misplaced import
            if TARGET in content:
                # But not at the top
                if not content.lstrip().startswith(TARGET):
                    fix_file(path)


if __name__ == "__main__":
    scan_and_fix()
