"""A 300-second CUDA retraining check; all writes go into a private disposable root.

First populate --root with reuse_stages.py. Its verified caches are read only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def worker(args):
    import numpy as np
    import torch
    from run_splat import TRAINING, Trial, camera, make_strategy
    from reuse_stages import checked_donor, validate_import
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; this is not the CPU --smoke renderer')
    donor, _, _ = checked_donor(args.root, args.sid)
    source_work = args.root / args.sid
    source_proof = args.root / 'send_back' / args.sid / 'proof'
    private = args.private_root
    work = private / args.sid
    proof = private / 'send_back' / args.sid / 'proof'
    work.mkdir(parents=True)
    proof.mkdir(parents=True)
    for name in ('depth', 'mesh_depth', 'nadir', 'initial.npz'):
        (work / name).symlink_to(source_work / name, target_is_directory=name != 'initial.npz')
    shutil.copy2(source_proof / 'scale.json', proof / 'scale.json')
    trial_args = argparse.Namespace(sid=args.sid, pack=args.pack, root=private, smoke=False,
                                    iterations=1105, points=donor['target_points'])
    trial = Trial(trial_args)
    validate_import(source_work, trial.config)
    with np.load(work / 'initial.npz') as initial:
        initial_count = len(initial['means'])
    tick = time.monotonic()
    trial.train_model(stop_after=1100)
    saved = torch.load(work / 'checkpoint.pt', map_location='cpu', weights_only=False)
    before = len(saved['params']['means'])
    if saved['step'] != 1100 or saved['sh_degree'] != 1 or saved['refine_events'] < 1 or before == initial_count:
        raise RuntimeError('Smoke did not cross SH1 and a real count-changing refinement')
    expected_scheduler = saved['scheduler']['last_epoch']
    expected_refines = saved['refine_events']
    del saved
    del trial
    torch.cuda.empty_cache()
    # A new Trial exercises shape-changing parameter construction and Adam,
    # strategy, exposure, scheduler and RNG restoration, not just torch.load.
    resumed = Trial(trial_args)
    resumed.train_model()
    final = torch.load(work / 'checkpoint.pt', map_location='cpu', weights_only=False)
    if final['step'] != 1105 or final['scheduler']['last_epoch'] != expected_scheduler + 5:
        raise RuntimeError('Resumed optimizer/scheduler did not continue for five steps')
    if len(final['params']['means']) != before or final['refine_events'] != expected_refines:
        raise RuntimeError('Resumed topology/strategy state mismatch')
    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return True
    if not finite(final):
        raise RuntimeError('Non-finite checkpoint parameters or optimizer/strategy state')
    if any(work.glob('*.done.json')):
        raise RuntimeError('Smoke unexpectedly wrote a stage marker')
    report = json.loads((proof / 'training.json').read_text())
    del final
    # Isolated reset/gradient probe: no checkpoint is overwritten and no fake
    # training steps are recorded. Probe the real callback at its reset boundary.
    saved = torch.load(work / 'checkpoint.pt', map_location='cuda', weights_only=False)
    params = {k: torch.nn.Parameter(v) for k, v in saved['params'].items()}
    opts = {k: torch.optim.Adam([v]) for k, v in params.items()}
    for k, opt in opts.items():
        opt.load_state_dict(saved['optimizers'][k])
    strategy, state = make_strategy(), saved['strategy_state']
    strategy.check_sanity(params, opts)
    pano, face = resumed.fixed_faces('train')[0]
    view = camera(pano, face)
    reset_step = TRAINING['strategy']['reset_every']
    rgb, _, _, info = resumed.render(params, view, 2048, 2048, with_info=True)
    strategy.step_pre_backward(params, opts, state, reset_step, info)
    rgb.square().mean().backward()
    if not all(v.grad is not None and finite(v.grad) for v in params.values()):
        raise RuntimeError('Missing or non-finite CUDA gradients')
    strategy.step_post_backward(params, opts, state, reset_step, info, packed=True)
    if float(params['opacities'].sigmoid().max()) > 2 * strategy.prune_opa + 1e-6:
        raise RuntimeError('Opacity reset callback failed')
    del rgb, info, params, opts, saved, state
    torch.cuda.empty_cache()
    # Offset checks use the unmodified checkpoint, with identity appearance.
    params = resumed.trained()
    offsets = []
    with torch.no_grad():
        for dx in (-.25, .25):
            offset = view.copy()
            offset[0, 3] += dx
            rgb, depth, alpha = resumed.render(params, offset, 2048, 2048)
            if not finite((rgb, depth, alpha)):
                raise RuntimeError('Non-finite render at an offset view')
            offsets.append({'camera_local_offset_metres': dx,
                            'alpha_coverage_above_.1': float((alpha > .1).float().mean())})
    del params, rgb, depth, alpha
    # Verify the actual SH1 PLY with the installed converter, without web assets.
    ply, spz = private / 'smoke.ply', private / 'smoke.spz'
    resumed.write_ply(ply)
    subprocess.run(['splat-transform', str(ply), str(spz)], check=True)
    if not spz.exists() or spz.stat().st_size == 0:
        raise RuntimeError('SH1 SPZ conversion produced no output')
    print(json.dumps({
        'steps': report['iterations'], 'resolution': report['resolution'],
        'seconds_per_step': report['seconds'] / report['iterations'],
        'wall_seconds_after_cache_verification': time.monotonic() - tick,
        'initial_gaussians': initial_count, 'checkpoint_gaussians': before,
        'gaussians': report['gaussians'], 'refine_events': report['refine_events'],
        'peak_vram_bytes': report['peak_gpu_allocated_bytes'],
        'train_PSNR': report['psnr_series']['train'][-1],
        'heldout_PSNR': report['psnr_series']['heldout'][-1],
        'SH1_SPZ_bytes': spz.stat().st_size, 'checkpoint_resume': '1100 -> 1105',
        'finite_gradients_and_state': True, 'reset_callback_probe_step': reset_step,
        'offset_views': offsets,
        'source_root_unchanged': str(args.root),
    }, indent=2), flush=True)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sid', choices=['VN4b1MGnnCR', 'HFwcH424n9w'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pack', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--private-root', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root, args.pack = args.root.expanduser().resolve(), args.pack.expanduser().resolve()
    if args.private_root:
        worker(args)
        return
    if not (args.root / args.sid / 'reuse_manifest.json').is_file():
        parser.error('--root must contain stages imported with reuse_stages.py')
    # Never construct a Trial at --root itself, even if a real root is supplied.
    with tempfile.TemporaryDirectory(prefix='hp-retrain-smoke-') as temp:
        command = [sys.executable, str(Path(__file__).resolve()), args.sid, '--root', str(args.root),
                   '--pack', str(args.pack), '--private-root', temp]
        child = subprocess.Popen(command, start_new_session=True)
        try:
            code = child.wait(timeout=max(.01, 300 - (time.monotonic() - started)))
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            parser.exit(124, 'CUDA smoke exceeded 300 seconds; killed worker and converter.\n')
        if code:
            parser.exit(code, 'CUDA smoke failed; no real-root stage markers were written.\n')


if __name__ == '__main__':
    main()
