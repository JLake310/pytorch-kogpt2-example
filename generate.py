from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import pandas as pd
from utils import generate


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("./best_model").to(device)
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")

    labels = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
    gen_sentences = {}
    gen_num = 10000

    for label in labels:
        print(f"Generating {label}...")
        gen_sentences[label] = generate(label, tokenizer, model, gen_num)

    label_nums = []
    for num in range(7):
        nums = [num] * gen_num
        label_nums += nums

    texts = []
    for label in labels:
        texts += gen_sentences[label]

    gen_data = pd.DataFrame({"target": label_nums, "text": texts})
    gen_data.to_csv("./gen_data.csv", index=False)
