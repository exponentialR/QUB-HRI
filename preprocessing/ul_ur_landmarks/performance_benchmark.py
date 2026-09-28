"""Bounded all-component throughput experiment; source frames and weights stay unchanged."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np

from .collection_backend import CollectionBackend
from .schema import sha256


def configure_sessions(backend, threads):
    """Experiment with ORT CPU pools; retain the original CUDA provider options."""
    import onnxruntime as ort
    for adapter in (backend.person, backend.pose, backend.hand_pose):
        model = adapter.model
        providers = [(name, options) for name, options in model.session.get_provider_options().items()]
        providers.sort(key=lambda p: p[0] != 'CUDAExecutionProvider')
        model.session = None
        gc.collect()
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        model.session = ort.InferenceSession(model.onnx_model, sess_options=options, providers=providers)
        if model.session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('CUDA provider required')


def compare(first, second):
    """Compare every field, including missingness, actor, topology and tracks."""
    result = {'structural_or_label_differences': 0, 'finite_mask_differences': 0,
              'max_abs_delta_by_field': {}}
    def visit(a, b, key):
        if isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or a.shape != b.shape:
                result['structural_or_label_differences'] += 1
                return
            good = np.isfinite(a) & np.isfinite(b)
            result['finite_mask_differences'] += int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))
            delta = float(np.max(np.abs(a[good] - b[good]))) if good.any() else 0.
            result['max_abs_delta_by_field'][key] = max(delta, result['max_abs_delta_by_field'].get(key, 0.))
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                result['structural_or_label_differences'] += 1
                return
            for name in a:
                visit(a[name], b[name], key + '.' + name)
        elif isinstance(a, (list, tuple)):
            if not isinstance(b, (list, tuple)) or len(a) != len(b):
                result['structural_or_label_differences'] += 1
                return
            for x, y in zip(a, b):
                visit(x, y, key)
        elif isinstance(a, float) and isinstance(b, float):
            result['max_abs_delta_by_field'][key] = max(abs(a-b), result['max_abs_delta_by_field'].get(key, 0.))
        elif isinstance(b, (np.ndarray, dict, list, tuple)) or a != b:
            result['structural_or_label_differences'] += 1
    visit([asdict(f) for f in first], [asdict(f) for f in second], 'frame')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('input-root', 'selection', 'models-dir', 'hand-weights', 'lego-weights', 'output-root'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--suite', choices=('threads','schedule'), default='threads')
    a = p.parse_args()
    a.output_root.mkdir(parents=True, exist_ok=False)
    import cv2
    import torch
    cv2.setNumThreads(2)
    selection = json.loads(a.selection.read_text())
    clips, sources = [], []
    for pair in (selection[0], selection[7]):
        for view, info in pair['views'].items():
            path = a.input_root / info['relpath']
            cap = cv2.VideoCapture(str(path))
            frames = []
            for _ in range(32):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Benchmark requires 32 decoded frames per selected clip')
                frames.append(frame)
            cap.release()
            clips.append(frames)
            sources.append({'path': str(path), 'sha256': sha256(path), 'view': view,
                            'frame_indices': list(range(32)), 'fps': info['fps']})
    with CollectionBackend(a.models_dir, a.lego_weights, a.hand_weights, a.output_root/'yolo_config',
            batch_size=8, cpu_workers=4, hand_mode='native_hand5', hand_joint_threshold=.2,
            hand_box_expansion=1.4, object_mode='full_lower', face_mode='crop_fallback',
            other_hand_nms_iou=.3, max_hands_per_actor=2) as backend:
        workers = backend.task_workers.copy()
        baseline = None
        report = {'sources': sources, 'model': backend.provenance, 'variants': [],
                  'code_sha256': sha256(Path(__file__)),
                  'limitations': '128 source frames; 3 warmed repeats; inference only, excludes initialization, video decoding and file writes. Full-run extrapolation requires end-to-end confirmation.'}
        variants=([(None,8,2,False),(1,8,2,False),(1,16,2,False),(1,32,2,False),(1,16,4,False),(2,16,4,False)]
                  if a.suite=='threads' else
                  [(None,8,2,False),(None,8,2,True),(None,8,4,True),(None,16,4,True),(None,8,4,False)])
        for threads, batch, count, overlap in variants:
            configure_sessions(backend, threads)
            backend.pool.shutdown(wait=True)
            backend.pool = ThreadPoolExecutor(max_workers=count)
            backend.task_workers = workers[:count]
            backend.batch_size = batch
            backend.overlap_face_tasks = overlap
            def run():
                outputs, timings = [], {}
                start = time.perf_counter()
                for images, source in zip(clips, sources):
                    backend.reset(images[0].shape[1], images[0].shape[0])
                    for offset in range(0, len(images), batch):
                        outputs.extend(backend.process_batch(images[offset:offset+batch],
                            [i*1000/source['fps'] for i in range(offset, min(offset+batch,len(images)))]))
                    for key, value in backend.timings.items():
                        timings[key] = timings.get(key,0.) + value
                torch.cuda.synchronize()
                return time.perf_counter()-start, outputs, timings
            run()  # Warm kernels and algorithm selection for this shape.
            trials = [run() for _ in range(3)]
            if baseline is None:
                baseline = trials[-1][1]
            row = {'ort_threads': threads, 'ort_spinning': threads is None, 'batch': batch,
                   'cpu_workers': count, 'overlap_face_tasks': overlap, 'seconds': [r[0] for r in trials],
                   'median_ms_per_frame': float(np.median([r[0] for r in trials])*1000/128),
                   'timings_last_repeat': trials[-1][2],
                   'output_comparison': compare(baseline,trials[-1][1])}
            report['variants'].append(row)
            (a.output_root/'benchmark.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
        backend.task_workers = workers  # Close every independently constructed task.


if __name__ == '__main__':
    main()
