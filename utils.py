import torch
from tqdm import tqdm


def generate(input_text, tokenizer, model, num):
    sentence_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_ids = tokenizer(input_text + "|", return_tensors="pt")["input_ids"].to(device)
    for cnt in tqdm(range(num)):
        gen_ids = model.generate(
            token_ids,
            max_length=32,
            repetition_penalty=2.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            use_cache=True,
            do_sample=True,
        )
        sentence = tokenizer.decode(gen_ids[0])
        sentence = sentence[sentence.index("|") + 1 :]
        if "<pad>" in sentence:
            sentence = sentence[: sentence.index("<pad>")].rstrip()
        sentence = sentence.replace("<unk>", " ").split("\n")[0]

        if cnt % 100 == 0 and cnt != 0:
            print(sentence)
        sentence_list.append(sentence)
    return sentence_list
