import json
import re
def is_valid_entry(line):
    try:
        data = json.loads(line)
        messages = data.get("messages", [])
        
        if len(messages) < 3:
            return False

        assistant_content = next((m["content"] for m in messages if m["role"] == "assistant"), "")


        match = re.search(r"<thinking>\s*Misconception:\s*(.*?)\s*</thinking>\s*(.*)", assistant_content, re.DOTALL)
        
        if not match:
            return False

        misconception_text = match.group(1).strip()
        hint_text = match.group(2).strip()

        if not misconception_text or not hint_text:
            return False

        return True
    except Exception:
        return False

output_filename = "./Data/llama3_sft_train.jsonl"
input_filename = "./Data/llama3_sft_tutor.jsonl"
valid_count = 0
removed_count = 0

with open(input_filename, "r") as f_in, open(output_filename, "w") as f_out:
    for line in f_in:
        if is_valid_entry(line):
            f_out.write(line)
            valid_count += 1
        else:
            removed_count += 1

print(f"Done! Kept: {valid_count} | Removed: {removed_count}")