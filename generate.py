from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import pandas as pd
from utils import generate
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_per_label", default=100, type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("./best_model").to(device)
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")

    labels = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]

    gen_sentences = {}
    for label in labels:
        print(f"Generating {label}...")
        gen_sentences[label] = generate(label, tokenizer, model, args.num_per_label)

    label_nums = []
    for num in range(7):
        nums = [num] * args.num_per_label
        label_nums += nums

    texts = []
    for label in labels:
        texts += gen_sentences[label]

    gen_data = pd.DataFrame({"target": label_nums, "text": texts})
    gen_data.to_csv("./gen_data.csv", index=False)
