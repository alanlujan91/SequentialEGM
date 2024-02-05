from __future__ import annotations

import glob
import re

# Get all .md files recursively
md_files = glob.glob("**/*.md", recursive=True)

for md_file in md_files:
    with open(md_file) as f:
        content = f.read()

    # The regular expression ([a-zA-Z\.])\n([a-zA-Z\.])
    # matches any letter (a-z or A-Z) or a period,
    # followed by a newline, and then another letter or a period.
    # The replacement string \1 \2 refers to the matched characters.
    # So this line replaces a newline between a letter or a period with a
    # space.
    new_content = re.sub(r"([a-zA-Z\.])\n([a-zA-Z\.])", r"\1 \2", content)

    # Write the modified content back to the file
    with open(md_file, "w") as f:
        f.write(new_content)
