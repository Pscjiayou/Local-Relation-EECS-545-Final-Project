import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models.vtsum_blip import v2vt_sum
from models.video_clip import build_video_clip_model
import utils
from utils import cosine_lr_schedule, update_config
from data import create_dataset, create_sampler, create_loader
from data.utils import collate_fn_padd_vtsum, collate_fn_padd_vtsum_eval


def train(model, data_loader, optimizer, device):
    model.train()
    print("🚀 Debug training loop started")
    for i, (video_embeddings, video_mask, vsum_labels, tsum_labels, _) in enumerate(data_loader):
        video_embeddings = video_embeddings.to(device)
        video_mask = video_mask.to(device)
        vsum_labels = vsum_labels.to(device)

        loss_tsum, loss_vsum = model(video_embeddings, video_mask, vsum_labels, tsum_labels)
        loss = loss_tsum + loss_vsum

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Batch {i}] Loss: {loss.item():.4f} | TSum: {loss_tsum.item():.4f} | VSum: {loss_vsum.item():.4f}")

        if i >= 1:
            print("🛑 Debug batch limit reached.")
            break


def main(args, config):
    print("⚙️ Debug mode active")
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    print("📂 Loading small dataset")
    train_dataset, _, _ = create_dataset('videoxum', config)
    sampler = None
    train_loader, _, _ = create_loader(
        [train_dataset, train_dataset, train_dataset], [sampler]*3,
        batch_size=[2, 2, 2], num_workers=[0, 0, 0],
        is_trains=[True, False, False],
        collate_fns=[collate_fn_padd_vtsum]*3)

    print("🧠 Creating model")
    if config['model'] == 'vtsum_blip_tt':
        model = v2vt_sum(config['model'], pretrained=config['pretrained'],
                        tt_depth=config['tt_depth'],
                        loss_type=config['loss_type'],
                        vit=config['vit'],
                        prompt=config['prompt'],
                        max_text_length=config['max_text_length'])

    elif config['model'] == 'vtsum_blip_tt_ca':
        model = v2vt_sum(config['model'], pretrained=config['pretrained'],
                        tt_depth=config['tt_depth'],
                        kernel_size=config['kernel_size'],  # 只有这个模型才用到 kernel_size
                        loss_type=config['loss_type'],
                        vit=config['vit'],
                        prompt=config['prompt'],
                        max_text_length=config['max_text_length'])

    else:
        raise ValueError(f"Unsupported model type: {config['model']}")


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)

    print("✅ Begin training")
    train(model, train_loader, optimizer, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/vtsum_blip.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config = update_config(config, args)

    main(args, config)
