# V8.3 Stage-2d M1 Stop-Threshold Migration Report

- Date: 2026-10-19
- Branch: `experiment/v83-spatial-foundation`
- Scope: one eval-only migration of the Stage-2d Stop threshold to the existing
  Transformer-M M1 static visual fusion checkpoint. No training, no
  architecture change, no M1 weight modification.
- Data: Dev Val 180 (Short/Medium/Long 60/60/60). Final Test 300
  (100/100/100) is sealed and not read.
- M1 checkpoint:
  `runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z/Transformer-M/M1/best.pt`
  (sha256 `6cac7883419ebeae172b6f0e9dc1ddf6b87e4c0dc7f8994bb3e185e044287b00`).
- Thresholds: 0.5 (Stage-2a reference) and 0.7 (Stage-2d safe rule).
- Every result is a direct mean over all 180 real closed-loop episodes; no
  weighting, no re-generation, no M0 re-run (M0@0.7 is read from the existing
  Stage-2d run).

## 1. M1@0.5 replay

M1@0.5 reproduces the Stage-2a M1 reference:

| Metric | Stage-2a M1 | M1@0.5 rerun |
|---|---:|---:|
| SR | 27.78% (50/180) | **27.78% (50/180)** |
| OSR | 30.56% | **30.56%** |
| nDTW | 0.3648 | **0.3648** |
| SPL | 0.2778 | 0.2778 |
| Premature-Stop | 55.00% | 55.00% |
| Timeout | 17.22% | 17.22% |
| Mean steps | 138.1 | 138.1 |

Outcome mismatches: **0/180** (success/OSR/premature/timeout). Four
numeric `min_distance_m` differences remain at 1e-5 to 1.3 m; they occur on
episodes whose outcome fields are identical and are due to CUDA
non-bit-reproducibility. The required SR/OSR/nDTW replay is exact, so the
migration proceeded.

## 2. M1@0.5 vs M1@0.7 — overall and per band

| Method | Band | SR | Success | OSR | SPL | nDTW | Premature | Timeout | Steps | Min dist m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1@0.5 | Overall | 27.78% | 50/180 | 30.56% | 0.2778 | 0.3648 | 55.00% | 17.22% | 138.1 | 132.7 |
| M1@0.5 | Short | 40.00% | 24/60 | 41.67% | 0.4000 | 0.5412 | 45.00% | 15.00% | 122.5 | 57.1 |
| M1@0.5 | Medium | 23.33% | 14/60 | 28.33% | 0.2333 | 0.3285 | 53.33% | 23.33% | 155.3 | 132.7 |
| M1@0.5 | Long | 20.00% | 12/60 | 21.67% | 0.2000 | 0.2247 | 66.67% | 13.33% | 136.5 | 208.5 |
| M1@0.7 | Overall | **31.11%** | **56/180** | **36.67%** | **0.3111** | **0.3684** | **47.22%** | 21.67% | 159.9 | 119.2 |
| M1@0.7 | Short | **43.33%** | **26/60** | **48.33%** | **0.4333** | 0.5313 | 38.33% | 18.33% | 140.8 | 51.2 |
| M1@0.7 | Medium | 23.33% | 14/60 | 33.33% | 0.2333 | 0.3313 | 48.33% | 28.33% | 181.6 | 113.8 |
| M1@0.7 | Long | **26.67%** | **16/60** | **28.33%** | **0.2667** | **0.2425** | **55.00%** | 18.33% | 157.4 | 192.5 |

Paired M1@0.7 vs M1@0.5 (bootstrap 95% CI over episodes; W/L are paired
improvements/regressions):

| Band | SR Δ (W/L, CI) | OSR Δ (W/L, CI) | Premature Δ (W/L, CI) | Timeout Δ (W/L, CI) | SPL Δ (CI) | nDTW Δ (CI) |
|---|---|---|---|---|---|---|
| Overall | **+3.33 pp [0.56, 6.67] (7W/1L)** | **+6.11 pp [2.78, 10.00] (11W/0L)** | **−7.78 pp [−11.67, −4.44] (14W/0L)** | +4.44 pp [1.67, 7.78] (0W/8L) | **+0.0333 [0.0056, 0.0667]** | +0.0036 [−0.0170, 0.0251] |
| Short | +3.33 pp [0.00, 8.33] (2W/0L) | +6.67 pp [1.67, 13.33] (4W/0L) | −6.67 pp [−13.33, −1.67] (4W/0L) | +3.33 pp [0.00, 8.33] (0W/2L) | +0.0333 [0.0000, 0.0833] | −0.0099 [−0.0516, 0.0256] |
| Medium | 0.00 pp [−5.00, 5.00] (1W/1L) | +5.00 pp [0.00, 11.67] (3W/0L) | −5.00 pp [−11.67, 0.00] (3W/0L) | +5.00 pp [0.00, 11.67] (0W/3L) | 0.0000 [−0.0500, 0.0500] | +0.0029 [−0.0388, 0.0476] |
| Long | **+6.67 pp [1.67, 13.33] (4W/0L)** | +6.67 pp [1.67, 13.33] (4W/0L) | **−11.67 pp [−20.00, −5.00] (7W/0L)** | +5.00 pp [0.00, 11.67] (0W/3L) | **+0.0667 [0.0167, 0.1333]** | **+0.0179 [0.0007, 0.0486]** |

**M1 threshold migration result:** threshold 0.7 transfers to M1. SR
+3.33 pp (7W/1L), OSR +6.11 pp (11W/0L), Premature −7.78 pp, SPL +0.0333,
nDTW +0.0036 (CI includes 0; not degraded overall; Long +0.0179 with a CI
above 0). The cost is Timeout +4.44 pp and reach-no-success 5→10/180; one
M1@0.5 success is lost. The gain is concentrated in Short and Long; Medium
SR is unchanged.

## 3. M1@0.7 vs existing M0@0.7

M0@0.7 is read from the existing Stage-2d R4 run; it was not re-run.

| Method | Band | SR | Success | OSR | SPL | nDTW | Premature | Timeout | Steps | Min dist m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0@0.7 | Overall | 34.44% | 62/180 | 38.33% | 0.3432 | 0.3573 | 48.89% | 16.67% | 147.9 | 126.4 |
| M0@0.7 | Short | 48.33% | 29/60 | 56.67% | 0.4833 | 0.5179 | 36.67% | 15.00% | 141.0 | 51.8 |
| M0@0.7 | Medium | 33.33% | 20/60 | 36.67% | 0.3312 | 0.3372 | 51.67% | 15.00% | 153.9 | 126.3 |
| M0@0.7 | Long | 21.67% | 13/60 | 21.67% | 0.2152 | 0.2169 | 58.33% | 20.00% | 148.8 | 201.3 |
| M1@0.7 | Overall | 31.11% | 56/180 | 36.67% | 0.3111 | **0.3684** | 47.22% | 21.67% | 159.9 | 119.2 |
| M1@0.7 | Short | 43.33% | 26/60 | 48.33% | 0.4333 | 0.5313 | 38.33% | 18.33% | 140.8 | 51.2 |
| M1@0.7 | Medium | 23.33% | 14/60 | 33.33% | 0.2333 | 0.3313 | 48.33% | 28.33% | 181.6 | 113.8 |
| M1@0.7 | Long | 26.67% | 16/60 | 28.33% | 0.2667 | 0.2425 | 55.00% | 18.33% | 157.4 | 192.5 |

Paired M1@0.7 vs M0@0.7:

| Band | SR Δ (W/L, CI) | OSR Δ (W/L, CI) | Premature Δ (W/L, CI) | Timeout Δ (W/L, CI) | SPL Δ (CI) | nDTW Δ (CI) |
|---|---|---|---|---|---|---|
| Overall | −3.33 pp [−10.00, 3.33] (16W/22L) | −1.67 pp [−8.33, 5.00] (17W/20L) | −1.67 pp [−9.44, 5.56] (25W/22L) | +5.00 pp [−2.22, 12.22] (17W/26L) | −0.0321 [−0.0978, 0.0340] | +0.0111 [−0.0389, 0.0626] |
| Short | −5.00 pp [−18.33, 8.33] (6W/9L) | −8.33 pp [−20.00, 3.33] (5W/10L) | +1.67 pp [−10.00, 13.33] (6W/7L) | +3.33 pp [−8.33, 15.00] (6W/8L) | −0.0500 [−0.1833, 0.0833] | +0.0134 [−0.0895, 0.1182] |
| Medium | **−10.00 pp [−21.67, 0.00] (3W/9L)** | −3.33 pp [−13.33, 6.67] (4W/6L) | −3.33 pp [−15.00, 8.33] (8W/6L) | **+13.33 pp [1.67, 26.67] (4W/12L)** | −0.0979 [−0.2125, 0.0042] | −0.0058 [−0.0953, 0.0827] |
| Long | +5.00 pp [−5.00, 16.67] (7W/4L) | +6.67 pp [−3.33, 18.33] (8W/4L) | −3.33 pp [−18.33, 11.67] (11W/9L) | −1.67 pp [−13.33, 10.00] (7W/6L) | +0.0515 [−0.0485, 0.1667] | +0.0256 [−0.0472, 0.0992] |

**M1 vs M0@0.7 result:** M1@0.7 has a higher nDTW point estimate
(0.3684 vs 0.3573, +0.0111) and a better Long band (SR +5.0 pp, nDTW
+0.0256), but lower SR (−3.33 pp), OSR (−1.67 pp), SPL (−0.0321), and
higher Timeout (+5.0 pp) overall. None of the overall paired differences is
statistically significant (all CIs include 0), but the Medium band is a
clear regression: SR −10.0 pp, SPL −0.0979, Timeout +13.33 pp (CI
[1.67, 26.67], significant). The apparent nDTW advantage is not
statistically significant overall (CI [−0.0389, 0.0626]).

## 4. Far/near Stop diagnostics

| Method | Band | Far-40 stop/step | Far-40 stops/ep | Boundary stops/ep | Near-20 stop rate | Near stops/steps | Reach-no-success | Stop median m | Far>100 stops |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1@0.5 | Overall | 0.360% | 0.472 | 0.078 | 44.2% | 50/113 | 5/180 | 103.5 | 76 |
| M1@0.7 | Overall | **0.274%** | **0.417** | **0.056** | 32.0% | 56/175 | 10/180 | 66.7 | **68** |
| M1@0.5 | Short | 0.318% | 0.350 | 0.100 | 63.2% | 24/38 | 1/60 | 20.5 | 16 |
| M1@0.7 | Short | 0.248% | 0.317 | 0.067 | 42.6% | 26/61 | 3/60 | 19.8 | 15 |
| M1@0.5 | Medium | 0.301% | 0.450 | 0.083 | 26.4% | 14/53 | 3/60 | 119.9 | 25 |
| M1@0.7 | Medium | 0.230% | 0.400 | 0.083 | 16.5% | 14/85 | 6/60 | 114.2 | 22 |
| M1@0.5 | Long | 0.460% | 0.617 | 0.050 | 54.5% | 12/22 | 1/60 | 213.3 | 35 |
| M1@0.7 | Long | 0.345% | 0.533 | 0.017 | 55.2% | 16/29 | 1/60 | 201.7 | 31 |

Interpretation:
- **Far-distance false Stops decrease.** Far-40 stop/step 0.360%→0.274%
  (−24%), far-40 stops/ep 0.472→0.417 (−11.7%), far>100 stops 76→68
  (−10.5%), boundary stops 0.078→0.056.
- **Near-goal missed Stops increase.** Reach-no-success 5→10/180,
  Timeout 17.22%→21.67% (+4.44 pp), and one M1@0.5 success is lost. The
  near-20 Stop rate falls 44.2%→32.0%, but near stops rise 50→56 because
  near exposure rises (113→175 steps). The real near cost is the 5 extra
  reach-no-success episodes.
- **nDTW does not degrade overall.** 0.3648→0.3684 (+0.0036, CI includes
  0); Long +0.0179 (CI above 0); Short −0.0099 (CI includes 0); Medium
  +0.0029 (CI includes 0).

## 5. Conclusion

**Threshold 0.7 transfers to M1, but M1 static visual fusion does not show
a reliable extra value over M0@0.7 in this Stop-threshold experiment.**

- The migration itself is successful: M1@0.7 vs M1@0.5 gives SR
  27.78→31.11 (+3.33 pp, 7W/1L), OSR 30.56→36.67 (+6.11 pp, 11W/0L),
  Premature 55.00→47.22 (−7.78 pp), SPL +0.0333, and nDTW +0.0036 (not
  degraded; Long +0.0179 significant). The price is Timeout +4.44 pp and
  reach-no-success 5→10, with one success lost.
- Against the existing M0@0.7, M1@0.7 is worse in SR (−3.33 pp), OSR
  (−1.67 pp), and SPL (−0.0321), and has more Timeout (+5.0 pp); its nDTW
  point estimate is higher (+0.0111) but not statistically significant.
  Medium is a clear regression (SR −10.0 pp, Timeout +13.33 pp, SPL
  −0.0979); Long is better (SR +5.0 pp, nDTW +0.0256). The overall
  trade-off does not establish extra value.
- **Recommendation: do not start M1 training based on this experiment.**
  Keep M0@0.7 as the Stop-threshold candidate. If the Long/nDTW signal is
  pursued, it needs a targeted M1 analysis (especially the Medium
  Timeout/SR regression) before any training; that is not started here.

## 6. Discipline and artifacts

- No training; only the existing Stage-2a M1 checkpoint was loaded.
- No M1 network architecture or weight was modified; checkpoint sha256 is
  unchanged.
- No M0 re-run; M0@0.7 is read from the existing Stage-2d R4 per-episode CSV.
- Final Test 300 is sealed and not read.
- All 180 Dev Val episodes are real closed-loop runs; metrics are direct
  means over all 180, no weighting.
- Run:
  `runs/v83_spatial_stage2/stage2d/m1_threshold_20261019T013737Z/`
  (`metrics.json`, `replay_m1_0.5.json`, `M1@0.5_per_episode.csv`,
  `M1@0.7_per_episode.csv`, `steps/`, `stops/`, `risk.csv`, `summary.md`,
  `provenance.json`).
- Code:
  `v83_spatial_stage2/stage2d_m1_threshold_eval.py`,
  `v83_spatial_stage2/stage2d_m1_summarize.py`,
  `v83_spatial_stage2/test_stage2d_m1.py`.
