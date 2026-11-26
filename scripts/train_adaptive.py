import os
import sys
import math
import yaml
import json
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.adaptive_hybrid_detector import AdaptiveHybridDetector
from src.losses.compute_budget_loss import ComputeBudgetLoss
from src.losses.retinanet_loss import RetinaNetLoss
from src.data.coco import RealCOCODataset, coco_train_transform, collate_fn
from train_resume import (ResumableRandomSampler, RunTracker, save_checkpoint_atomic,
                          rng_state, set_rng_state, already_finished)

def main():
    parser = argparse.ArgumentParser(description="Train Adaptive CNN-Swin Detector")
    parser.add_argument('--config', type=str, default='configs/adaptive_hybrid.yaml', help='Path to config file')
    parser.add_argument('--max_batches', type=int, default=None, help='Max batches per epoch for diagnostic/debug runs')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs from config')
    parser.add_argument('--resume', action='store_true', help='Save last.pth periodically and resume from it if present')
    parser.add_argument('--ckpt_every', type=int, default=2000, help='Steps between resume checkpoints (with --resume)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    torch.manual_seed(config['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training: {config['model']['name']}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Epochs: {epochs}, Max batches per epoch: {args.max_batches if args.max_batches else 'Full dataset'}")
    print(f"Budget target: {config['training']['budget_target']}, Budget weight: {config['training']['budget_weight']}")
    print(f"{'='*60}\n")

    dataset = RealCOCODataset(
        root=config['data']['train_dir'],
        annFile=config['data']['train_ann'],
        transform=coco_train_transform
    )
    batch_size = config['data']['batch_size']
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    sampler = ResumableRandomSampler(len(dataset), config['training']['seed']) if args.resume else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    model = AdaptiveHybridDetector(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained_backbone']
    ).to(device)

    det_loss_fn = RetinaNetLoss(num_classes=config['model']['num_classes']).to(device)
    budget_loss_fn = ComputeBudgetLoss(target_budget=config['training']['budget_target']).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['mixed_precision'])

    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    resume_path = os.path.join(output_dir, 'last.pth')
    tracker = RunTracker(
        progress_path="results/progress_adaptive.json",
        model_name=config['model']['name'], epochs=epochs,
        steps_per_epoch=steps_per_epoch if args.max_batches is None else min(steps_per_epoch, args.max_batches),
        loss_keys=['total', 'det', 'budget'])
    start_epoch, start_step = 0, 0
    if args.resume and already_finished(tracker.progress_path, resume_path, os.path.join(output_dir, 'final_model.pth')):
        print("Training already finished earlier (final_model.pth saved). Nothing to do.", flush=True)
        return

    final_p3_dist = None
    final_p4_dist = None

    if args.resume and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location='cpu', weights_only=False)
        if ck.get('model_name') == config['model']['name'] and ck.get('epochs') == epochs:
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            scheduler.load_state_dict(ck['scheduler_state_dict'])
            scaler.load_state_dict(ck['scaler_state_dict'])
            set_rng_state(ck.get('rng'))
            tracker.load_state_dict(ck['tracker'])
            final_p3_dist = ck.get('final_p3_dist')
            final_p4_dist = ck.get('final_p4_dist')
            start_epoch, start_step = ck['epoch'], ck['step_in_epoch']
            tracker.resumed_from = {'epoch': start_epoch + 1, 'step': start_step, 'at': time.time()}
            tracker.last_ckpt = ck.get('saved_info')
            print(f"Resumed from {resume_path}: epoch {start_epoch + 1}, step {start_step}/{steps_per_epoch}", flush=True)
            del ck
        else:
            print(f"Ignoring {resume_path}: it belongs to a different model or epoch count", flush=True)

    def save_resume(epoch_idx, step_in_epoch):
        info = {'epoch': epoch_idx + 1, 'step': step_in_epoch, 'at': time.time()}
        save_checkpoint_atomic({
            'model_name': config['model']['name'],
            'epochs': epochs,
            'epoch': epoch_idx,
            'step_in_epoch': step_in_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'rng': rng_state(),
            'tracker': tracker.state_dict(),
            'final_p3_dist': final_p3_dist,
            'final_p4_dist': final_p4_dist,
            'saved_info': info,
        }, resume_path)
        tracker.last_ckpt = info
        print(f"[checkpoint] saved {resume_path} at epoch {epoch_idx + 1}, step {step_in_epoch}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        offset = start_step if epoch == start_epoch else 0
        if offset == 0:
            tracker.reset_epoch()
        if sampler is not None:
            sampler.set_epoch(epoch, offset * batch_size)

        for batch_idx, (images, targets) in enumerate(dataloader, start=offset):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            images = torch.stack(images).to(device)
            targets = [{k: (v.to(device) if hasattr(v, 'to') else v) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                cls_logits, bbox_preds, w_p3, w_p4 = model(images, tau=config['training']['tau'])
                cls_loss, box_loss = det_loss_fn(cls_logits, bbox_preds, targets)
                det_loss = cls_loss + box_loss

                b_loss = budget_loss_fn(w_p3) + budget_loss_fn(w_p4)
                total_loss = det_loss + config['training']['budget_weight'] * b_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0).item()
            scaler.step(optimizer)
            scaler.update()

            step_done = batch_idx + 1
            tracker.add_step({'total': total_loss.item(), 'det': det_loss.item(), 'budget': b_loss.item()},
                             epoch * tracker.steps_per_epoch + step_done)

            final_p3_dist = w_p3.mean(dim=(0, 2, 3)).detach().cpu().numpy().tolist()
            final_p4_dist = w_p4.mean(dim=(0, 2, 3)).detach().cpu().numpy().tolist()

            if step_done % 10 == 0 or (args.max_batches is not None and step_done == args.max_batches):
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step_done}/{steps_per_epoch if args.max_batches is None else args.max_batches}] | Total: {total_loss.item():.4f} (Det: {det_loss.item():.4f}, Budget: {b_loss.item():.4f}) | GradNorm: {grad_norm:.2f} | LR: {lr:.2e}", flush=True)
            if args.resume:
                if step_done % 50 == 0:
                    tracker.write('training', epoch, step_done, {
                        'last': {'total': total_loss.item(), 'det': det_loss.item(), 'budget': b_loss.item()},
                        'grad_norm': grad_norm, 'lr': optimizer.param_groups[0]['lr'],
                        'loss_scale': scaler.get_scale(),
                        'p3_dist': final_p3_dist, 'p4_dist': final_p4_dist})
                if step_done % args.ckpt_every == 0 and step_done < tracker.steps_per_epoch:
                    save_resume(epoch, step_done)

        scheduler.step()
        avg = tracker.epoch_averages()

        tracker.history.append({
            'epoch': epoch + 1,
            'total_loss': avg['total'],
            'det_loss': avg['det'],
            'budget_loss': avg['budget'],
            'p3_dist': final_p3_dist,
            'p4_dist': final_p4_dist,
            'steps': tracker.steps,
            'nan_steps': tracker.nan_steps
        })
        print(f"--> Epoch {epoch+1} Summary: Avg Total = {avg['total']:.4f} (Det: {avg['det']:.4f}, Budget: {avg['budget']:.4f}) | P3 Dist: {[round(x, 3) for x in final_p3_dist]} | P4 Dist: {[round(x, 3) for x in final_p4_dist]}\n", flush=True)
        if args.resume:
            tracker.reset_epoch()
            save_resume(epoch + 1, 0)
            tracker.write('training', epoch + 1, 0)

    total_time = tracker.elapsed()
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    params = sum(p.numel() for p in model.parameters())
    history = tracker.history

    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'seed': config['training']['seed'],
        'history': history,
        'total_time_s': round(total_time, 2),
        'max_memory_mb': round(max_memory, 2),
        'params': params,
        'final_p3_dist': final_p3_dist,
        'final_p4_dist': final_p4_dist
    }
    ckpt_path = os.path.join(output_dir, 'final_model.pth')
    save_checkpoint_atomic(checkpoint, ckpt_path)
    print(f"\nTraining Complete! Saved validated checkpoint to {ckpt_path}")

    metrics = {
        "Model": config['model']['name'],
        "Params": params,
        "GPU_memory_mb": round(max_memory, 2),
        "total_time_s": round(total_time, 2),
        "P3_dist": final_p3_dist,
        "P4_dist": final_p4_dist,
        "history": history
    }
    with open('results/adaptive_training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    if args.resume:
        tracker.write('done', epochs - 1, tracker.steps_per_epoch)
        if os.path.exists(resume_path):
            os.remove(resume_path)

if __name__ == '__main__':
    main()
