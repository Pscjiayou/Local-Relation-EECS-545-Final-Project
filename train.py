import torch
import utils
import os
import csv
from torch.utils.tensorboard import SummaryWriter


class EarlyStopper:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta

    def check(self, current_score):
        if self.best_score is None or current_score > self.best_score + self.delta:
            self.best_score = current_score
            self.counter = 0
            return False  # do not stop
        else:
            self.counter += 1
            return self.counter >= self.patience


def train(model, data_loader, optimizer, epoch, device, args, writer=None, early_stopper=None):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train Epoch: [{epoch}]'
    print_freq = args.print_freq

    for i, (video_embeddings, video_mask, vsum_labels, tsum_labels, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)):

        video_embeddings = video_embeddings.to(device)
        video_mask = video_mask.to(device)
        vsum_labels = vsum_labels.to(device)

        loss_tsum, loss_vsum = model(video_embeddings, video_mask, vsum_labels, tsum_labels)
        loss = args.lambda_tsum * loss_tsum + args.lambda_vsum * loss_vsum

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_tsum=loss_tsum.item())
        metric_logger.update(loss_vsum=loss_vsum.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # 记录日志
    metric_logger.synchronize_between_processes()
    logs = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", logs)

    # TensorBoard 写入
    if writer:
        for key, value in logs.items():
            writer.add_scalar(key, value, epoch)

    # CSV 日志写入
    log_path = os.path.join(args.output_dir, 'train.log.csv')
    first_write = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer_csv = csv.writer(f)
        if first_write:
            writer_csv.writerow(['epoch'] + list(logs.keys()))
        writer_csv.writerow([epoch] + [f"{logs[k]:.4f}" for k in logs])

    # Early stopping 检查
    if early_stopper:
        if early_stopper.check(-logs['loss']):  # 用负的 loss 判断 early stop
            print(f"⛔ Early stopping at epoch {epoch}")
            return True  # stopped
    return False  # continue
