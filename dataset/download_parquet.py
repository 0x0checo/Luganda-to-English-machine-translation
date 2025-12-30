from datasets import load_dataset
from huggingface_hub import login


# Paste your token here
 login()

dataset = load_dataset("parquet", data_files="https://huggingface.co/datasets/Sunbird/salt/resolve/main/text-all/test-00000-of-00001.parquet")

print(dataset["train"])  # look at the first sample

# Select only the two columns
subset = dataset["train"].remove_columns(
    [c for c in dataset["train"].column_names if c not in ["eng_source_text", "lug_text"]]
)

# Write each column to its own text file
with open("eng.txt", "w", encoding="utf-8") as f_eng, open("lug.txt", "w", encoding="utf-8") as f_lug:
    for example in subset:
        f_eng.write(example["eng_source_text"] + "\n")
        f_lug.write(example["lug_text"] + "\n")
