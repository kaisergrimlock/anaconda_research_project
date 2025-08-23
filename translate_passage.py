import re
import json
import random
from pathlib import Path

input_path = "outputs/topic_and_docs_q524332.txt"
output_path = "outputs/topic_and_docs_q524332_modified.txt"
insert_phrase = input("Enter the phrase to insert randomly into passages: ")

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    content = infile.read()

    # Find all Passage JSON blocks
    def insert_randomly(text, phrase):
        if not text:
            return phrase
        idx = random.randint(0, len(text))
        return text[:idx] + phrase + text[idx:]

    def replace_passage(match):
        passage_json = match.group(1)
        passage = json.loads(passage_json)
        passage["contents"] = insert_randomly(passage["contents"], insert_phrase)
        return f'Passage:\n{json.dumps(passage, ensure_ascii=False, indent=2)}\n'

    # Substitute each Passage block with the modified one
    modified_content = re.sub(
        r'Passage:\n({.*?})\n',
        replace_passage,
        content,
        flags=re.DOTALL
    )

    outfile.write(modified_content)

print(f"Modified file written to: {output_path}")