import os
from datetime import datetime

header_template = """\
# ============================================================
# File: {filename}
# Purpose: {purpose}
# Version: {version}
# Author: Your Team
# Date: {date}
# ============================================================
"""


def add_header_to_script(script_path, purpose, version="1.2"):
    """Adds the header to the script if it's not already present."""
    date = datetime.now().strftime("%B %Y")
    filename = os.path.basename(script_path)
    header = header_template.format(
        filename=filename, purpose=purpose, version=version, date=date
    )

    with open(script_path, "r", encoding="utf-8") as file:
        content = file.read()

    if (
        "# File:" not in content.splitlines()[0:5]
    ):  # check first few lines for existing header
        with open(script_path, "w", encoding="utf-8") as file:
            file.write(header + "\n" + content)
        print(f"Header added to {script_path}")
    else:
        print(f"Header already present in {script_path}")


def process_directory(directory, purpose="Automated header insertion", version="1.2"):
    """Walk through all files in the given directory and insert header into Python files."""
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                script_path = os.path.join(subdir, file)
                add_header_to_script(script_path, purpose, version)


# Run
process_directory("src")
