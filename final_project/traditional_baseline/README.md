# Traditional baseline: RGB decision tree

This is a classical supervised pixel classifier, not a DNN or graph-cut method.
It predicts one of the same six Potsdam classes from raw R, G, B values.
A decision tree is suitable here because it is inexpensive to fit on CPU and
provides a reproducible reference without installing a graph-cut optimizer.
It has no spatial/context features, so expect noisy regions and ambiguity
between vegetation classes. Performance must be measured, not guessed.

## Kaggle

1. Upload `traditional_baseline` as a code dataset. Attach the SAME final Potsdam
   data already used for the deep models; do not upload the imagery again.
2. Import `kaggle_demo.ipynb`. No GPU is required.
3. Run path discovery. If multiple matches appear, set CODE_SOURCE and DATA_ROOT
   explicitly. DATA_ROOT can be a manifest filename or its parent folder.
4. Run dependency checks. Internet is needed only if a missing package must be installed.
5. Run training/search, then validation. These produce real full-tile metrics.
6. Set RUN_TEST=True and run the test cell after fixing the selected configuration.
7. Save outputs/download the archive. For comparison, provide training/search.json,
   validation/report.json and test/report.json.

## Experimental protocol

- Original manifest and seed 20261110; exclude training tiles 4_12 and 6_7,
  matching the deep runs. Expected effective split: 16 train / 6 validation / 14 test.
- Uniform random sampling without replacement: 30,000 valid pixels per training
  tile (480,000 total). This is a fixed resource budget, not all 576 million pixels.
- Raw RGB features; no normalisation, augmentation, class reweighting or TTA.
  Trees do not need RGB standardisation. The classical baseline deliberately has
  its own training procedure; the deep-model identical-pipeline requirement does
  not mean a tree can be trained with AdamW.
- Automated exhaustive search over max_depth={12,20} and
  min_samples_leaf={20,100}; all four candidates fit the same sampled pixels.
  Objective: pooled six-class validation mIoU over every validation pixel.
  Stop after four trials, retain the first candidate on exact ties.
  Search bounds and sampling budget are declared experimental choices; they are
  not themselves claimed to have been agent-optimised.
- Test labels are opened only in explicit test evaluation. The selected model
  remains trained on training data only. Do not refit or tune using test results.
- Reports include pooled confusion matrix, per-class IoU/F1/precision/recall,
  macro means, pixel accuracy and exact-match macro boundary F1. Undefined
  class metrics are null and excluded from their macro mean.
- Boundary F1 uses the same implementation as the existing deep validation
  measurements: four-neighbour, class-aware inner boundaries, zero-pixel
  tolerance; exclude exterior/ignored interfaces. It is not tolerant boundary F1.
- Timings are CPU measurements, not GPU-comparable model throughput. Model-only
  time covers predict calls, excluding RGB conversion and loading. End-to-end
  tile time includes reading, feature conversion, metrics and optional writes.
- Run with several GB of available RAM; one full tile is loaded at a time and
  predictions are calculated in row strips. Four validation sweeps are required
  during search plus one final validation and one test sweep. Runtime is measured
  and logged, not promised in advance.

## Local commands

```text
python -m pip install -r traditional_baseline/requirements.txt
python traditional_baseline/run.py train --data final_dataset --output baseline_results/training
python traditional_baseline/run.py evaluate --data final_dataset --checkpoint baseline_results/training/best.joblib --split validation --output baseline_results/validation
python traditional_baseline/run.py evaluate --data final_dataset --checkpoint baseline_results/training/best.joblib --split test --output baseline_results/test
```

Only load your own trusted joblib checkpoints. The manifest hash is checked at
evaluation. Reopening a session does not require retraining if best.joblib was
saved and attached. Set CHECKPOINT to its attached path.

For the report call the method **RGB decision tree (CART)**, not graph cut.
Only compare validation with validation, and test with test. JSON scores are
fractions; multiply by 100 for percentage tables. FLOPs and neural parameter
counts are not reported for the tree. Four candidate fits constitute one
baseline search experiment, not four new deep-model runs.
