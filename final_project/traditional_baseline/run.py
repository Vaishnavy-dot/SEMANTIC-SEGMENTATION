"""CPU RGB decision tree; validation-only grid search and held-out evaluation."""
import argparse
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import rasterio
import sklearn
from sklearn.tree import DecisionTreeClassifier
from boundary import counts, CLASSES

PALETTE = np.array([(255,255,255),(0,0,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0)], dtype=np.uint8)
EXCLUDED = {'4_12', '6_7'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def decode(values, record):
    if values.dtype != np.uint8:
        raise ValueError('Masks must be uint8')
    if record.get('label_encoding', 'rgb') == 'class_ids':
        if values.shape[0] != 1 or not np.isin(values, [0,1,2,3,4,5,255]).all():
            raise ValueError('Invalid class-ID mask')
        return values[0]
    if values.shape[0] != 3:
        raise ValueError('Expected RGB mask')
    result = np.full(values.shape[1:], 255, dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        result[np.all(values == color[:,None,None], axis=0)] = i
    unknown = result == 255
    if record.get('ignore_black', False):
        unknown &= np.any(values != 0, axis=0)
    if unknown.any():
        raise ValueError(f"Unknown label colours in {record['tile_id']}")
    return result


def records(root, split):
    manifest = json.loads((root/'manifest.json').read_text())
    rows = manifest['tiles']
    ids = [r['tile_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate tile IDs: possible split leakage')
    selected = []
    index = None
    for row in rows:
        if row['tile_id'] in EXCLUDED:
            if row['split'] != 'train':
                raise ValueError('Excluded tile is outside training split')
            continue
        if row['split'] != split:
            continue
        row = dict(row)
        for key in ('image', 'mask'):
            expected = root / row[key]
            if not expected.is_file():
                if index is None:
                    index = {}
                    for p in root.rglob('*'):
                        if p.suffix.lower() in ('.tif', '.tiff'):
                            index.setdefault(p.name.lower(), []).append(p)
                matches = index.get(Path(row[key]).name.lower(), [])
                if len(matches) != 1:
                    raise FileNotFoundError(f'{expected}: found {len(matches)} filename matches')
                expected = matches[0]
            row[key] = expected
        selected.append(row)
    if not selected:
        raise ValueError(f'No {split} tiles')
    return selected


def read_tile(record):
    with rasterio.open(record['image']) as src:
        if src.count != 3 or set(src.dtypes) != {'uint8'}:
            raise ValueError('Expected uint8 RGB image')
        rgb, profile = src.read(), src.profile
    with rasterio.open(record['mask']) as src:
        truth = decode(src.read(), record)
    if rgb.shape[1:] != truth.shape:
        raise ValueError('Image/mask dimensions differ')
    return rgb, truth, profile


def metrics(cm, boundary):
    tp = np.diag(cm).astype(float)
    reference, predicted = cm.sum(1), cm.sum(0)
    def divide(n, d):
        return np.divide(n, d, out=np.full(6, np.nan), where=d != 0)
    scores = dict(iou=divide(tp, reference+predicted-tp),
                  f1=divide(2*tp, reference+predicted),
                  precision=divide(tp,predicted), recall=divide(tp,reference))
    per_class = {name: {k: float(v[i]) if np.isfinite(v[i]) else None for k,v in scores.items()}
                 for i,name in enumerate(CLASSES)}
    result = {'confusion_matrix':cm.tolist(), 'per_class':per_class,
              'valid_pixels':int(cm.sum()), 'pixel_accuracy':float(tp.sum()/cm.sum())}
    result.update({'mean_'+k:float(np.nanmean(v)) for k,v in scores.items()})
    if boundary is not None:
        p,g,t = boundary
        bf = divide(2*t,p+g)
        result['boundary_f1'] = float(np.nanmean(bf))
        result['boundary_counts_predicted_reference_matched'] = boundary.tolist()
        result['boundary_per_class_f1'] = {name:float(bf[i]) if np.isfinite(bf[i]) else None for i,name in enumerate(CLASSES)}
        result['boundary_protocol'] = 'Exact same-class inner boundaries; 4-neighbour; zero-pixel tolerance; ignored interfaces and image exterior excluded; pooled class counts then macro average'
    return result


def evaluate(model, rows, out=None):
    start = time.perf_counter()
    cm = np.zeros((6,6), dtype=np.int64)
    bc = np.zeros((3,6), dtype=np.int64) if out else None
    timings = []
    for record in rows:
        tile_start = time.perf_counter()
        rgb, truth, profile = read_tile(record)
        pred = np.empty(truth.shape, dtype=np.uint8)
        model_seconds = 0
        for y in range(0,truth.shape[0],128):
            features = np.ascontiguousarray(rgb[:,y:y+128].transpose(1,2,0).reshape(-1,3), dtype=np.float32)
            tick = time.perf_counter()
            labels = model.predict(features)
            model_seconds += time.perf_counter()-tick
            pred[y:y+128] = labels.reshape(pred[y:y+128].shape)
            target = truth[y:y+128]
            valid = target != 255
            cm += np.bincount(target[valid].astype(np.int64)*6 + pred[y:y+128][valid], minlength=36).reshape(6,6)
        if out:
            bc += counts(pred,truth)
            dest = out/'predictions'
            dest.mkdir(parents=True,exist_ok=True)
            profile.update(count=1,dtype='uint8',nodata=255,compress='lzw')
            with rasterio.open(dest/f"top_potsdam_{record['tile_id']}_prediction.tif",'w',**profile) as dst:
                dst.write(pred,1)
        timings.append(dict(tile_id=record['tile_id'], model_only_seconds=model_seconds,
                            end_to_end_seconds=time.perf_counter()-tile_start))
        print(record['tile_id'], 'completed', flush=True)
    result = metrics(cm,bc)
    result.update(tiles=len(rows), tile_timings=timings, elapsed_seconds=time.perf_counter()-start,
                  timing_note='CPU model-only excludes feature conversion/loading; end-to-end includes loading, conversion, metrics and output writing when enabled; no patch stitching required')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train','evaluate'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    parser.add_argument('--seed',type=int,default=20261110)
    parser.add_argument('--samples-per-tile',type=int,default=30000)
    args = parser.parse_args()
    root = args.data.resolve()
    if root.is_file():
        root = root.parent
    digest = hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode == 'train':
        if args.samples_per_tile < 1:
            raise ValueError('samples-per-tile must be positive')
        rng = np.random.default_rng(args.seed)
        xs,ys = [],[]
        train_rows = records(root,'train')
        for record in train_rows:
            rgb,truth,_ = read_tile(record)
            valid = np.flatnonzero(truth.ravel()!=255)
            chosen = rng.choice(valid, min(len(valid),args.samples_per_tile),replace=False)
            xs.append(rgb.reshape(3,-1)[:,chosen].T)
            ys.append(truth.ravel()[chosen])
            print('Sampled',record['tile_id'],len(chosen),flush=True)
        x,y = np.concatenate(xs),np.concatenate(ys)
        del xs,ys,rgb,truth,valid
        if set(np.unique(y)) != set(range(6)):
            raise ValueError('Training sample does not cover all six classes')
        validation = records(root,'validation')
        best = -1
        trials = []
        for depth,leaf in itertools.product([12,20],[20,100]):
            model = DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=args.seed)
            start = time.perf_counter()
            model.fit(x,y)
            fit_seconds = time.perf_counter()-start
            report = evaluate(model,validation)
            params = dict(max_depth=depth,min_samples_leaf=leaf)
            trials.append(dict(parameters=params,fit_seconds=fit_seconds,validation=report))
            if report['mean_iou'] > best:
                best = report['mean_iou']
                joblib.dump(dict(model=model,manifest_sha256=digest,excluded_training_tiles=sorted(EXCLUDED),
                                 parameters=params,seed=args.seed,samples_per_tile=args.samples_per_tile,
                                 sample_counts=np.bincount(y,minlength=6).tolist(),features=['R','G','B'],
                                 sklearn_version=sklearn.__version__,python=platform.python_version()),args.output/'best.joblib')
            save(args.output/'search.json',dict(objective='Maximise full-validation six-class mIoU',
                 strategy='Exhaustive four-candidate grid; first candidate wins exact ties',budget=4,
                 stopping_rule='All four candidates evaluated',trials=trials,best_validation_miou=best))
        print('Selected checkpoint:',args.output/'best.joblib',flush=True)
    else:
        if not args.checkpoint:
            raise ValueError('--checkpoint is required; load only your own trusted joblib file')
        checkpoint = joblib.load(args.checkpoint)
        if checkpoint['manifest_sha256'] != digest:
            raise ValueError('Manifest differs from training')
        result = evaluate(checkpoint['model'],records(root,args.split),args.output)
        save(args.output/'report.json',dict(model='RGB decision tree',split=args.split,
             parameters=checkpoint['parameters'],seed=checkpoint['seed'],manifest_sha256=digest,
             excluded_training_tiles=sorted(EXCLUDED),prediction={'overall':result}))
        print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
