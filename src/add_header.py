import os
from datetime import datetime

# Template for header
header_template = """
# ============================================================
# File: {filename}
# Purpose: {purpose}
# Version: 1.2
# Author: Your Team
# Date: {date}
# ============================================================
"""


def add_header_to_script(script_path, purpose):
    """
    Adds the header to the script if it's not already present.
    """
    date = datetime.now().strftime("%B %Y")
    filename = os.path.basename(script_path)
    header = header_template.format(filename=filename, purpose=purpose, date=date)

    # Read the content of the script
    with open(script_path, 'r') as file:
        content = file.read()

    # If the header is already present, do nothing
    if not content.startswith(
        "# =========================================================="):
        with open(script_path, 'w') as file:
            # Prepend the header and then the rest of the content
            file.write(header + content)


def process_directory(directory, purpose="Automated header insertion"):
    """
    Walk through all files in the given directory and insert header into Python files.
    """
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                script_path = os.path.join(subdir, file)
                add_header_to_script(script_path, purpose)


# Specify the root directory where your scripts are located
process_directory("src")  # You can change this to any other directory if needed
