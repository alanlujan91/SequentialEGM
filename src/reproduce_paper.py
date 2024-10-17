from __future__ import annotations

import glob

# Get 'egmn.md' file
egmn_file = "content/paper/egmn.md"

# Read 'egmn.md' and keep only the metadata
with open(egmn_file) as f:
    egmn_content = f.read().split("---\n")[1]

# Get all .md files that start with a number
md_files = sorted(glob.glob("content/paper/[0-9]*.md"))

# Read and process all files using a list comprehension
merged_content = (
    "---\n"
    + egmn_content
    + "---\n\n"
    + "\n\n".join(open(md_file).read().split("---\n")[-1] for md_file in md_files)
)

# Overwrite 'egmn.md' with merged content
with open(egmn_file, "w") as f:
    f.write(merged_content)
