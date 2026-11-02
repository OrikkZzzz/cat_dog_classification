import torch

from .meters import AverageMeter


def _autocast(device, amp):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp and device.type == "cuda",
    )


def set_bn_eval(model):
    #冻结时BN的running stats不更新
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, *,
                    amp=True, bn_eval=False, log_interval=50, logger=print, scheduler=None):
    model.train()
    if bn_eval:
        set_bn_eval(model)

    loss_meter, acc_meter = AverageMeter(), AverageMeter()

    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            output = model(images)
            loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        correct = (output.argmax(dim=1) == labels).sum().item()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update_sum(correct, batch_size)

        if log_interval and step % log_interval == 0:
            logger(
                f"  ep{epoch} [{step}/{len(loader)}] "
                f"loss {loss_meter.avg:.4f} acc {acc_meter.avg:.4f}"
            )

    return loss_meter.sync().avg, acc_meter.sync().avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, amp=True, collect=False):
    model.eval()

    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    preds, targets = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _autocast(device, amp):
            output = model(images)
            loss = criterion(output, labels)

        batch_size = labels.size(0)
        pred = output.argmax(dim=1)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update_sum((pred == labels).sum().item(), batch_size)

        if collect:
            preds.append(pred.cpu())
            targets.append(labels.cpu())

    if collect:
        return loss_meter.sync().avg, acc_meter.sync().avg, torch.cat(preds), torch.cat(targets)
    return loss_meter.sync().avg, acc_meter.sync().avg
