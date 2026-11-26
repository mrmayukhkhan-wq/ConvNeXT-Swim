"""Checkpoint/resume and live-progress helpers for long training runs.

Used by train_fixed_hybrid.py and train_adaptive.py when --resume is passed:
  * ResumableRandomSampler: per-epoch seeded shuffle, so a run can restart
    mid-epoch and continue with exactly the remaining samples of that epoch.
  * RunTracker: epoch accumulators, smoothed loss curve, speed estimate and an
    atomically written progress JSON that the dashboard reads.
  * save_checkpoint_atomic: write to a temp file, then rename, so a crash during
    saving never corrupts the previous checkpoint.
"""
import json
import math
import os
import time

import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler):
    def __init__(self, data_len, seed):
        self.data_len = data_len
        self.seed = seed
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch, start_index=0):
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.data_len, generator=g).tolist()
        return iter(perm[self.start_index:])

    def __len__(self):
        return self.data_len - self.start_index


def save_checkpoint_atomic(state, path):
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def write_json_atomic(path, data, retries=5):
    """Best effort: a reader holding the file open on Windows can block the rename."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.2)
    return False


class RunTracker:
    def __init__(self, progress_path, model_name, epochs, steps_per_epoch, loss_keys, point_every=200):
        self.progress_path = progress_path
        self.model_name = model_name
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.loss_keys = list(loss_keys)
        self.point_every = point_every
        self.history = []
        self.loss_points = []
        self.elapsed_before = 0.0
        self.session_start = time.time()
        self.resumed_from = None
        self.last_ckpt = None
        self.sec_per_step = None
        self._last_t = None
        self._last_gs = None
        self.reset_epoch()
        self._reset_window()

    def reset_epoch(self):
        self.sums = {k: 0.0 for k in self.loss_keys}
        self.steps = 0
        self.nan_steps = 0

    def _reset_window(self):
        self._win = {k: 0.0 for k in self.loss_keys}
        self._win_n = 0

    def add_step(self, values, global_step):
        # Keep the original averaging (a NaN step still makes the epoch average NaN),
        # but count bad steps and keep them out of the plotted curve.
        bad = any(not math.isfinite(v) for v in values.values())
        if bad:
            self.nan_steps += 1
        for k in self.loss_keys:
            self.sums[k] += values[k]
        self.steps += 1
        if not bad:
            for k in self.loss_keys:
                self._win[k] += values[k]
            self._win_n += 1
        if self._win_n >= self.point_every:
            self.loss_points.append([global_step] + [round(self._win[k] / self._win_n, 5) for k in self.loss_keys])
            self._reset_window()

    def epoch_averages(self):
        return {k: self.sums[k] / max(1, self.steps) for k in self.loss_keys}

    def elapsed(self):
        return self.elapsed_before + (time.time() - self.session_start)

    def write(self, status, epoch, step, info=None):
        now = time.time()
        global_step = epoch * self.steps_per_epoch + step
        if self._last_t is not None and global_step > self._last_gs:
            sps = (now - self._last_t) / (global_step - self._last_gs)
            self.sec_per_step = sps if self.sec_per_step is None else 0.9 * self.sec_per_step + 0.1 * sps
        self._last_t, self._last_gs = now, global_step
        data = {
            'model': self.model_name,
            'status': status,
            'epoch': min(epoch + 1, self.epochs),
            'epochs': self.epochs,
            'step': step,
            'steps_per_epoch': self.steps_per_epoch,
            'global_step': global_step,
            'total_steps': self.epochs * self.steps_per_epoch,
            'sec_per_step': self.sec_per_step,
            'elapsed_s': round(self.elapsed(), 1),
            'epoch_avg_so_far': self.epoch_averages(),
            'epoch_steps_so_far': self.steps,
            'nan_steps_epoch': self.nan_steps,
            'loss_keys': self.loss_keys,
            'loss_points': self.loss_points,
            'history': self.history,
            'last_checkpoint': self.last_ckpt,
            'resumed_from': self.resumed_from,
            'pid': os.getpid(),
            'updated_at': now,
        }
        if info:
            data.update(info)
        write_json_atomic(self.progress_path, data)

    def state_dict(self):
        return {
            'sums': self.sums, 'steps': self.steps, 'nan_steps': self.nan_steps,
            'history': self.history, 'loss_points': self.loss_points,
            'elapsed_s': self.elapsed(),
        }

    def load_state_dict(self, d):
        self.sums = d['sums']
        self.steps = d['steps']
        self.nan_steps = d.get('nan_steps', 0)
        self.history = d['history']
        self.loss_points = d.get('loss_points', [])
        self.elapsed_before = d['elapsed_s']
        self.session_start = time.time()


def already_finished(progress_path, resume_path, final_path):
    """True when a previous --resume run completed (supervisor died before recording it)."""
    if os.path.exists(resume_path) or not os.path.exists(final_path):
        return False
    try:
        with open(progress_path) as f:
            return json.load(f).get('status') == 'done'
    except (OSError, ValueError):
        return False


def rng_state():
    s = {'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        s['cuda'] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    if not s:
        return
    torch.set_rng_state(s['torch'])
    if 'cuda' in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])
