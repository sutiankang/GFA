import torch
import math
import sys
import random
import torch.nn.functional as F

from metrics.eval_jaccard import db_eval_iou_multi
from utils.distributed import is_main_process, reduce_value
from utils.utils import SmoothedValue, MetricLogger, get_total_grad_norm


def train_one_epoch(args, logger, model, criterion, train_loader, optimizer, device,
                    epoch, writer_dict, scheduler, model_ema, scaler):

    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: Epoch: [{}/{}]'.format(epoch, args.epochs)

    for batch in metric_logger.log_every(train_loader, args.print_freq, logger, header):
        image, flow, mask = batch['image'], batch['flow'], batch['mask']
        image, flow, mask = image.to(device), flow.to(device), mask.to(device)

        h, w = image.shape[-2:]
        if args.multi_scale:
            scale = random.sample(args.scales, 1)
            if scale != 1.0:
                h = int(round(h * scale / 32) * 32)
                w = int(round(w * scale / 32) * 32)
                image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=True)
                flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=True)
                mask = F.interpolate(mask, size=(h, w), mode="nearest")

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image, flow)
            # output no sigmoid
            losses = criterion(output, mask)
        loss_value = reduce_value(losses, average=True).item()

        if not math.isfinite(loss_value) and is_main_process():
            logger.info("Loss is {}, stopping training".format(loss_value))
            logger.info(loss_value)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if args.max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        else:
            grad_total_norm = get_total_grad_norm(model.parameters(), args.max_norm)
        # mask 0-1, output need 0-1
        precise_output = torch.sigmoid(output).cpu().detach().numpy()
        precise_output[precise_output >= 0.5] = 1
        precise_output[precise_output < 0.5] = 0

        iou = db_eval_iou_multi(mask.cpu().detach().numpy(), precise_output)
        iou = reduce_value(torch.tensor(iou, device=device), average=True).item()
        scheduler.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        metric_logger.update(mIoU=iou)

        if args.tensorboard and is_main_process():
            writer_dict["writer"].add_scalar("train_loss", metric_logger.meters["loss"].global_avg,
                                             writer_dict["train_global_steps"])
            writer_dict["writer"].add_scalar("learning_rate", metric_logger.meters["lr"].global_avg,
                                             writer_dict["train_global_steps"])
            writer_dict["writer"].add_scalar("mean_iou", metric_logger.meters["mIoU"].global_avg,
                                             writer_dict["train_global_steps"])
            writer_dict['train_global_steps'] += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if is_main_process():
        logger.info("Averaged stats: {}".format(metric_logger))


@torch.no_grad()
def valid_one_epoch(args, logger, model, criterion, val_loader, device, epoch, writer_dict):
    torch.cuda.empty_cache()
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test: Epoch: [{}/{}]'.format(epoch, args.epochs)
    for batch in metric_logger.log_every(val_loader, args.print_freq, logger, header):
        image, flow, mask = batch['image'], batch['flow'], batch['mask']
        image, flow, mask = image.to(device), flow.to(device), mask.to(device)

        output = model(image, flow)
        losses = criterion(output, mask)
        loss_value = reduce_value(losses, average=True).item()
        # mask 0-1 output need 0-1
        precise_output = torch.sigmoid(output).cpu().detach().numpy()
        precise_output[precise_output >= 0.5] = 1
        precise_output[precise_output < 0.5] = 0

        iou = db_eval_iou_multi(mask.cpu().detach().numpy(), precise_output)
        iou = reduce_value(torch.tensor(iou, device=device), average=True).item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(mIoU=iou)

        if args.tensorboard and is_main_process():
            writer_dict["writer"].add_scalar("valid_loss", metric_logger.meters["loss"].global_avg,
                                             writer_dict["val_global_steps"])
            writer_dict["writer"].add_scalar("mean_iou", metric_logger.meters["mIoU"].global_avg,
                                             writer_dict["val_global_steps"])
            writer_dict['val_global_steps'] += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    torch.cuda.synchronize()

    if is_main_process():
        logger.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}