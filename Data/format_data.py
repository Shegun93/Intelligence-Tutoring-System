from langchain_ollama import ChatOllama
from datasets import load_dataset
import json
import logging

logging.basicConfig(level=logging.INFO)

Judge_model = ChatOllama(
    model="qwen3:4b",
    temperature=0.8,
    validate_model_on_init=True,
    num_ctx=2048,
)

def get_diagnostic_map(item):
    """
    Send MCQ to Judge model and get a JSON map of misconceptions and hints for incorrect options.
    """
    prompt = f"""
You are an expert educator. Analyze this MCQ and its correct explanation.
QUESTION: {item['question']}
OPTIONS: {json.dumps(item['options'])}
CORRECT OPTION: {item['correct_option']}
EXPLANATION: {item['explanation']}

TASK: For each INCORRECT option, provide:
1. "misconception": Why would a student pick this? (Max 15 words)
2. "hint": A Socratic nudge to guide them to the right answer. (Max 25 words)
Return ONLY a JSON object where keys are the option letters (A, B, C, D, etc).
"""
    response = Judge_model.invoke(prompt)
    return json.loads(response.content)


data = load_dataset("json", data_files="Data/Physics_questions.json", split="train")

refactor = []


for idx, item in enumerate(data):
    try:
        diag_map = get_diagnostic_map(item)

        for opt_letter, opt_text in item['options'].items():
            if opt_letter == item['correct_option']:
                continue 

            diag = diag_map.get(opt_letter, {})
            sft_turn = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI Intelligent Tutoring System. Analyze the error in <thinking> tags, then provide a hint."
                    },
                    {
                        "role": "user",
                        "content": f"Q: {item['question']}\nMy Answer: {opt_letter}"
                    },
                    {
                        "role": "assistant",
                        "content": f"<thinking>\nMisconception: {diag.get('misconception', '')}\n</thinking>\n{diag.get('hint', '')}"
                    }
                ]
            }
            refactor.append(sft_turn)

        if (idx + 1) % 10 == 0:
            logging.info(f"Processed {idx + 1} questions")

    except Exception as e:
        logging.warning(f"Skipping item {idx} due to error: {e}")
        continue

output_file = "llama3_sft_tutor.jsonl"
with open(output_file, "w") as f:
    for entry in refactor:
        f.write(json.dumps(entry) + "\n")

logging.info(f"Saved {len(refactor)} JSONL entries to {output_file}")
