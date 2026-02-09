from langchain_ollama import ChatOllama
from datasets import load_dataset
import json


df = load_dataset("json", data_files="./Data/formatted_test.jsonl", split="train")


# def format_test(example):
#     return (f"Question: {example['question']}\n"
#             f"A. {example['options']['A']}\n"
#             f"B. {example['options']['B']}\n"
#             f"C. {example['options']['C']}\n"
#             f"D. {example['options']['D']}\n"
#             "Answer: "
#             )


#formatted_questions = df.map(lambda x: {"formatted": format_test(x)})

#formatted_questions = formatted_questions.remove_columns([col for col in formatted_questions.column_names if col != "formatted"])

#formatted_questions.to_json("./Data/formatted_test.jsonl")

#print("Formatted questions saved to ./Data/formatted_test.jsonl")

print(df["formatted"][3])


