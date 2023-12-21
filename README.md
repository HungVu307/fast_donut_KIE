# DOcument Understanding Transformer with FastViT encoder on CORD dataset.
This project was carried out while working at VinBigdata
My code was based on [Original Repo](https://github.com/clovaai/donut/tree/master)
## Theory
We experiment with Donut model, by replace Swin-Transformer encoder by FastViT encoder:
![plot](image/1.png)
New model only has 81M params and with re-params technique, we can train on multi-branch and reference on sigle-branch. More information can be found at [FastViT](https://arxiv.org/abs/2303.14189)
## Installation
First, git clone this repo
``` bash
git clone https://github.com/HungVu307/fast_donut_KIE
cd donut_project
```
## For training
To train with CORD dataset:
```bash
python train.py
```
You can modify config at 'config/train_cord.yaml'

## For testing
To test with CORD dataset:
```bash
python test.py --pretrained_model_name_or_path 'YOUR_PATH'
```