# July 10 CFKD Reliability Experiment Summary

Branch: `idea/reliability-local-cf-distill-20260710`

This run evaluates the minimal Reliability-Calibrated Local CF Distillation implementation with two stage-2 SimCSE checkpoints:

- `Qwen2-0.5B-CFKD-uncalibrated`: local CF distillation without reliability weighting.
- `Qwen2-0.5B-CFKD-reliability`: local CF distillation with reliability weighting.

Both checkpoints reached `checkpoint-1000`. Due to shared-GPU OOM pressure, steps 500-1000 used the low-memory setting `cf_distill_anchor_subsample=16` for both variants, so the comparison is controlled but should be treated as a pilot result.

## Training Summary

| Model | Global step | Last-100 SimCSE loss | Last-100 CF KD loss | Last-100 total loss | Last-100 CF weight mean |
|---|---:|---:|---:|---:|---:|
| Uncalibrated CFKD | 1000 | 0.008276 | 3.296173 | 0.173084 | 1.000000 |
| Reliability CFKD | 1000 | 0.012468 | 3.528116 | 0.188874 | 0.029991 |

No NaN was observed in the completed training logs.

## Downstream SASRec Results

Metrics are mean over the repeated downstream evaluation seeds.

| Dataset | Uncal N@20 | Rel N@20 | Delta N@20 | Uncal R@20 | Rel R@20 | Delta R@20 |
|---|---:|---:|---:|---:|---:|---:|
| Games_5core | 0.054341 | 0.050370 | -0.003970 | 0.108878 | 0.101394 | -0.007483 |
| Arts_5core | 0.066547 | 0.063160 | -0.003386 | 0.106334 | 0.106007 | -0.000327 |
| Movies_5core | 0.044162 | 0.042890 | -0.001272 | 0.081379 | 0.081990 | +0.000611 |
| Sports_5core | 0.101895 | 0.093336 | -0.008559 | 0.128443 | 0.124616 | -0.003827 |
| Baby_5core | 0.053763 | 0.049931 | -0.003832 | 0.080933 | 0.081171 | +0.000238 |
| Goodreads | 0.091481 | 0.089102 | -0.002380 | 0.184009 | 0.180562 | -0.003446 |

## Conclusion

The current reliability weighting did not improve the downstream ranking metrics. `NDCG@20` dropped on all six datasets. `Recall@20` improved slightly on Movies_5core and Baby_5core, but the gains were very small and did not offset the consistent NDCG degradation.

The most likely issue is that the current reliability scale is too aggressive: the reliability-weighted run had an average CF weight near `0.03`, so the CF teacher signal was mostly suppressed. A better next experiment is to retune the reliability transform or normalize weights to preserve the average KD strength while still down-weighting low-confidence items.

## Result Files

Raw small text outputs are stored under `Results/*/SASRec/.../results.txt` and `config.json`. Large artifacts such as `output/`, `item_info/*.npy`, `seqrec/ckpt/*.pth`, and `run_logs/` are intentionally not tracked.
