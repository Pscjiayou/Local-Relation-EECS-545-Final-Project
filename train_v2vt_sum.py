import argparse
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vtsum_blip import v2vt_sum
from models.video_clip import build_video_clip_model
import utils
from utils import cosine_lr_schedule, update_config, compute_f1, compute_kendall, compute_spearman, concat_all_gather
from data import create_sampler, create_loader
from data import create_dataset
from data.utils import save_result, collate_fn_padd_vtsum, collate_fn_padd_vtsum_eval
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from train import train, EarlyStopper
from torch.utils.tensorboard import SummaryWriter


from modules import (
    AdaptiveFrameSampler,
    WhisperASR,
    ASRDenoiser,
    DPPSelector,
    smooth_summary,
    MemoryAugmentedTransformer
)
from torchvision import transforms
from PIL import Image

from eval_v2vt_sum import evaluate

def main(args, config):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating videoxum dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('videoxum', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset, val_dataset, test_dataset], [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None] * 3

    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset], samplers,
        batch_size=[config['batch_size']]*3, num_workers=[4, 4, 4],
        is_trains=[True, False, False],
        collate_fns=[collate_fn_padd_vtsum, collate_fn_padd_vtsum_eval, collate_fn_padd_vtsum_eval])

    print("Creating model")
    print("Model from config:", config['model'])

    base_model = v2vt_sum(
        config['model'], pretrained=config['pretrained'],
        tt_depth=config['tt_depth'],
        kernel_size=config.get('kernel_size', 5),
        loss_type=config['loss_type'],
        vit=config['vit'],
        prompt=config['prompt'],
        max_text_length=config['max_text_length'])

    model = MemoryAugmentedTransformer(base_model).to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    clip_model = build_video_clip_model("ViT-B/16", model_path=config['pretrained_clip'], device=args.device)
    clip_model_without_ddp = clip_model
    if args.distributed:
        clip_model = torch.nn.parallel.DistributedDataParallel(clip_model, device_ids=[args.gpu])
        clip_model_without_ddp = clip_model.module

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    start_epoch = 0
    # TensorBoard Writer 和 EarlyStopper 实例
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb'))
    early_stopper = EarlyStopper(patience=5)

    if args.resume:
        ckpt_path = os.path.join(args.output_dir, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            print(f"\U0001f501 Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model_without_ddp.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0) + 1
        else:
            print(f"Checkpoint not found at {ckpt_path}, cannot resume.")

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            stop = train(model, train_loader, optimizer, epoch, device, args, writer, early_stopper)
            if stop:
                print(f"⛔ Early stopping triggered at epoch {epoch}")
                break


        if utils.is_main_process():
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch
            }, os.path.join(args.output_dir, f'checkpoint.pth'))
        if args.distributed:
            dist.barrier()

        if (epoch + 1) % config['max_epoch'] == 0 or args.evaluate:
            evaluate(model_without_ddp, clip_model_without_ddp, val_loader, device, config, 'val')
            evaluate(model_without_ddp, clip_model_without_ddp, test_loader, device, config, 'test')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='vtsum_blip_tt_ca')
    parser.add_argument('--config', default='configs/vtsum_blip_tt_ca.yaml')
    parser.add_argument('--output_dir', default='outputs/vtsum_blip_tt_ca')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use for training')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--init_lr', type=float, default=1e-5)
    parser.add_argument('--max_epoch', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lambda_tsum', type=float, default=1.0)
    parser.add_argument('--lambda_vsum', type=float, default=1.0)
    parser.add_argument('--clip_root', type=str, default='dataset/ActivityNet/feat/vt_clipscore')

    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--ckpt_freq', type=int, default=28)

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', default=False, type=bool)

    args = parser.parse_args()

    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config = update_config(config, args)

    config['result_dir'] = os.path.join(args.output_dir, 'result')
    config['ckpt_dir'] = os.path.join(args.output_dir, 'checkpoints')
    config['logger_pth'] = os.path.join(args.output_dir, 'train.log')
    config['kernel_size'] = args.kernel_size
    args.logger_pth = config['logger_pth']

    Path(config['result_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['ckpt_dir']).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)
