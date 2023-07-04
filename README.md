# PyTorch-KoGPT2-example
base model : [skt/kogpt2-base-v2](https://github.com/SKT-AI/KoGPT2)

blog : [[PyTorch, KoGPT2] Fine-tuning하고 문장 생성하기(w/ full code)](https://velog.io/@k0310kjy/PyTorch-KoGPT2-Fine-tuning%ED%95%98%EA%B3%A0-%EB%AC%B8%EC%9E%A5-%EC%83%9D%EC%84%B1%ED%95%98%EA%B8%B0w-full-code)
## Fine-tuning
```bash
python train.py --batch_size 32 --epochs 10 --lr 2e-5 --warmup_steps 200
```

## Generate
```bash
python generate.py --num_per_label 10
```
 
