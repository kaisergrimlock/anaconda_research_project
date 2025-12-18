from pathlib import Path
import re

p = Path("vi.txt")
lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

# matches "( ... )" (non-greedy), same line
pattern = re.compile(r"\([^)]*\)")

lines_with_parens = 0
total_paren_segments = 0

for line in lines:
    matches = pattern.findall(line)
    if matches:
        lines_with_parens += 1
        total_paren_segments += len(matches)

print(f"Lines containing '(...)': {lines_with_parens}")
print(f"Total '(...)' segments found: {total_paren_segments}")
