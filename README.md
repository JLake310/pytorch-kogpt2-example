# PyTorch-KoGPT2-example
base model : [skt/kogpt2-base-v2](https://github.com/SKT-AI/KoGPT2)
## Fine-tuning
```bash
python train.py --batch_size 32 --epochs 10 --lr 2e-5 --warmup_steps 200
```

## Generate
```bash
python generate.py --num_per_label 10
```
 
