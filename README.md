# Teaching math to a small LLM with post-training

This project tries to teach a small language model grade-school math. The model
is Qwen2.5-0.5B-Instruct, the problems are GSM8K word problems, and all training
runs on one Apple Silicon laptop. Before training, the model's greedy answer is
correct on 0.477 of the test problems, and 0.368 of its sampled answers are
correct.

The project uses only post-training, with two ingredients: the model's own
sampled answers, and a checker that compares the final number with the GSM8K
answer key. There is no stronger teacher model and no human-written solutions.
The questions are how much math the model learns this way, whether the gain holds
on problems it never trained on, and which post-training method teaches it best.
Every method gets the same model, data and checker:

- **SFT**: fine-tune on the model's own correct answers.
- **DPO**: train on (correct, incorrect) answer pairs for the same question.
- **GRPO**: online RL; sample answers during training, reward 1 if correct, 0 if not.
- **RLHF with a learned reward model**: a reward model is trained on the DPO
  pairs, then used as the reward for GRPO or PPO in place of the checker.
- **Combinations**: DPO and GRPO started from the SFT model (SFT→DPO, SFT→GRPO).

## What the model learned

Test split, 300 GSM8K problems never used for training or for choosing settings.
Intervals are 95% paired bootstrap intervals over questions.

- Every method except PPO raises pass@1, the mean accuracy of 8 sampled answers.
  SFT→DPO raises it most, from 0.368 to 0.508 (+0.140 [+0.115, +0.165]).
- The gain holds on SVAMP, a second set of math word problems that was not used
  for training or selection. pass@1 rises by +0.062 to +0.117 for every method
  except PPO, and SFT→DPO is again highest.
- Greedy accuracy rises less, by +0.023 to +0.047 on test, and no interval
  excludes zero at n=300.
- pass@8 (at least one of 8 samples correct) rises significantly only for
  SFT→DPO, from 0.757 to 0.820. Depending on the method, 69% to 82% of the pass@1
  gain comes from questions where at least one of the base model's 8 samples was
  already correct. So post-training here seems to make the model more reliable on
  problems it could already solve some of the time, more than it teaches it to
  solve new ones. The experiments do not isolate this, and with 8 samples
  "could already solve" is itself measured with noise.
- Some methods stop writing the requested `Answer: <number>` line. Counting only
  answers that have it, SFT→DPO falls below base, and PPO and SFT→GRPO score
  highest (see Answer format below).

Comparing the methods:

- Starting from SFT helps. DPO and GRPO started from the SFT model beat the same
  methods started from base on pass@1 (test: DPO +0.074 [+0.051, +0.097],
  GRPO +0.026 [+0.003, +0.049]).
- SFT→DPO has the highest pass@1 on dev, test and SVAMP. On top of SFT, DPO
  adds +0.057 [+0.034, +0.080] test pass@1 (SVAMP +0.048 [+0.022, +0.073]), and
  GRPO adds +0.014 [-0.010, +0.037]. On test, no method improves greedy accuracy
  over SFT.
- Plain SFT on the model's own correct answers trains in 0.4 h and raises test
  pass@1 by +0.084, 60% of the SFT→DPO gain.
- PPO with a learned reward model did not beat base. The first reward model
  scored truncated answers above complete ones, so PPO learned to stop before
  finishing. A reward model trained with truncated-answer pairs fixed the
  truncation, and PPO still did not beat base.

All results are one seed per method with short LoRA runs; see Limitations.

## Approach

All methods use the same correctness signal: an exact-match check of the final
number against the GSM8K answer (`is_correct` in `posttrain/common.py`).
Training data comes from the base model itself: 8 sampled answers per training
question, graded by that check.

- SFT trains on the correct base-model answers.
- DPO trains on (correct, incorrect) pairs for the same question.
- GRPO samples during training and uses the 0/1 grade as reward.
- The reward model is trained on the DPO pairs; GRPO and PPO can use it in place
  of the grader.

LoRA (rank 16) is used unless `--full` is given.

Splits: train 1000 and valid 100 are used to build training data. Dev (300, from
the GSM8K train set, disjoint from train/valid) is used to compare runs and
choose settings. Test (300, from the GSM8K test set) is evaluated once at the
end. SVAMP (300 problems, a different set of math word problems) is used only to
test whether the gains carry over.

Metrics: greedy is one deterministic answer per question. pass@1 is the mean
accuracy of 8 answers sampled at temperature 0.8; pass@8 is the fraction of
questions where at least one of the 8 is correct. Best-of-8 (RM) is the accuracy
when the reward model `rm_trunc2` picks one of the 8 samples.

## Results

Qwen2.5-0.5B-Instruct, LoRA rank 16, one seed per method, n=300 questions per
split. All differences are paired: both models answer the same questions, and the
95% interval comes from resampling questions (`compare.py`, 10,000 draws).

Methods: SFT on correct base-model samples; DPO (lr 5e-6) on correct/incorrect
pairs; SFT→DPO is DPO started from the merged SFT model; GRPO with the exact-match
grader as reward; GRPO (RM) with reward model v1 as reward; SFT→GRPO is
grader-reward GRPO started from SFT; PPO (RM) with `rm_trunc2_q` as reward.
Reward model v1 is trained on the DPO pairs only. `rm_trunc2` and `rm_trunc2_q`
also get the truncated-answer pairs and differ only in training length (2 and
0.5 epochs).

### Gain over the base model

![difference from base, GSM8K test and SVAMP](results/plots/gains.png)

Each dot is a method's accuracy minus the base model's, with its 95% interval.
pass@1 (orange) rises for every method except PPO, on GSM8K test and on SVAMP.
Greedy (blue) and pass@8 (green) move less, and most of their intervals cross zero.

### What each method adds on top of SFT

![difference from SFT, GSM8K test and SVAMP](results/plots/vs_sft.png)

The same comparison with the SFT model as the reference. DPO started from SFT
adds +0.057 [+0.034, +0.080] pass@1 on test and +0.048 [+0.022, +0.073] on SVAMP.
GRPO started from SFT adds +0.014 [-0.010, +0.037] on test. On test, no method
improves greedy accuracy over SFT. DPO, GRPO and GRPO (RM) started from base are
level with SFT on test pass@1 (intervals include zero) and took longer to train
(1.3 to 10.8 h, against 0.4 h for SFT; see Wall-clock).

### Answer format

![pass@1 and strict pass@1, GSM8K test](results/plots/format_test.png)

The prompt asks for a final line `Answer: <number>`. pass@1 (orange) accepts the
last number in the answer when that line is missing; strict pass@1 (purple)
counts only complete answers that end with the line and are correct. DPO,
SFT→DPO and GRPO (RM) write the line less often than base, so their strict pass@1
falls below base. PPO and SFT→GRPO have the highest strict pass@1. Within each
method except PPO and GRPO (RM), answers with and without the line are about
equally accurate (difference 0.00 to 0.05), so the pass@1 gains do not come from
the fallback parser.

### PPO

![PPO KL](results/plots/ppo_kl.png)

The first PPO run used reward model v1, which scores truncated answers above
complete ones (`probe_rm.py`). KL reached about 21, 90% of dev answers hit the
token limit, and dev greedy accuracy fell to 0.137. With `rm_trunc2_q` KL stayed
near 3.5 and 0.5% of answers hit the limit, but pass@1 did not beat base (test
-0.029 [-0.053, -0.005]).

### All tables

<details>
<summary>Full tables for test, dev and SVAMP, answer format, wall-clock, and detailed findings</summary>

For best-of-8, `bon_rm.py` saves whether the picked sample is correct for each
question, and `compare.py` bootstraps that difference the same way.

#### Test (read once, no selection on it)

| method | greedy | Δ greedy vs base [95% CI] | pass@1 | Δ pass@1 vs base [95% CI] | pass@8 | Δ pass@8 vs base [95% CI] | best-of-8 (RM) | Δ best-of-8 vs base [95% CI] |
|---|---|---|---|---|---|---|---|---|
| base | 0.477 |  | 0.368 |  | 0.757 |  | 0.583 |  |
| SFT | 0.520 | +0.043 [-0.010, +0.097] | 0.452 | +0.084 [+0.060, +0.107] | 0.793 | +0.037 [-0.007, +0.080] | 0.613 | +0.030 [-0.023, +0.083] |
| DPO | 0.517 | +0.040 [-0.013, +0.090] | 0.435 | +0.067 [+0.043, +0.090] | 0.787 | +0.030 [-0.017, +0.073] | 0.593 | +0.010 [-0.047, +0.067] |
| SFT→DPO | 0.510 | +0.033 [-0.027, +0.090] | 0.508 | +0.140 [+0.115, +0.165] | 0.820 | +0.063 [+0.020, +0.107] | 0.617 | +0.033 [-0.020, +0.087] |
| GRPO (grader) | 0.520 | +0.043 [-0.010, +0.097] | 0.439 | +0.071 [+0.049, +0.094] | 0.780 | +0.023 [-0.023, +0.067] | 0.573 | -0.010 [-0.060, +0.040] |
| GRPO (RM) | 0.500 | +0.023 [-0.030, +0.077] | 0.450 | +0.083 [+0.057, +0.108] | 0.770 | +0.013 [-0.030, +0.057] | 0.580 | -0.003 [-0.057, +0.050] |
| SFT→GRPO | 0.523 | +0.047 [-0.010, +0.103] | 0.465 | +0.098 [+0.072, +0.122] | 0.797 | +0.040 [-0.007, +0.083] | 0.597 | +0.013 [-0.037, +0.063] |
| PPO (RM) | 0.427 | -0.050 [-0.110, +0.007] | 0.339 | -0.029 [-0.053, -0.005] | 0.710 | -0.047 [-0.100, +0.007] | 0.510 | -0.073 [-0.130, -0.017] |

#### Dev (used to choose settings)

| method | greedy | Δ greedy vs base [95% CI] | pass@1 | Δ pass@1 vs base [95% CI] | pass@8 | Δ pass@8 vs base [95% CI] | best-of-8 (RM) | Δ best-of-8 vs base [95% CI] |
|---|---|---|---|---|---|---|---|---|
| base | 0.527 |  | 0.444 |  | 0.827 |  | 0.697 |  |
| SFT | 0.587 | +0.060 [+0.010, +0.110] | 0.549 | +0.105 [+0.081, +0.129] | 0.893 | +0.067 [+0.030, +0.103] | 0.753 | +0.057 [+0.007, +0.107] |
| DPO | 0.607 | +0.080 [+0.027, +0.137] | 0.529 | +0.085 [+0.061, +0.109] | 0.870 | +0.043 [+0.010, +0.077] | 0.680 | -0.017 [-0.070, +0.037] |
| SFT→DPO | 0.627 | +0.100 [+0.040, +0.160] | 0.613 | +0.169 [+0.141, +0.197] | 0.910 | +0.083 [+0.050, +0.117] | 0.753 | +0.057 [+0.007, +0.110] |
| GRPO (grader) | 0.593 | +0.067 [+0.017, +0.120] | 0.522 | +0.078 [+0.055, +0.102] | 0.867 | +0.040 [+0.003, +0.077] | 0.683 | -0.013 [-0.067, +0.040] |
| GRPO (RM) | 0.630 | +0.103 [+0.050, +0.160] | 0.545 | +0.101 [+0.077, +0.126] | 0.860 | +0.033 [+0.000, +0.067] | 0.710 | +0.013 [-0.040, +0.067] |
| SFT→GRPO | 0.630 | +0.103 [+0.050, +0.157] | 0.564 | +0.120 [+0.095, +0.145] | 0.893 | +0.067 [+0.030, +0.103] | 0.750 | +0.053 [+0.000, +0.107] |
| PPO (RM) | 0.567 | +0.040 [-0.023, +0.107] | 0.435 | -0.009 [-0.034, +0.016] | 0.827 | +0.000 [-0.040, +0.040] | 0.663 | -0.033 [-0.083, +0.017] |

#### Transfer: SVAMP (not used for training or selection)

SVAMP test split, 300 problems (`prepare_transfer.py`), same prompt, grader and
sampling as above. SVAMP was added after the GSM8K results were known; no setting
was chosen on it.

| method | greedy | Δ greedy vs base [95% CI] | pass@1 | Δ pass@1 vs base [95% CI] | pass@8 | Δ pass@8 vs base [95% CI] | best-of-8 (RM) | Δ best-of-8 vs base [95% CI] |
|---|---|---|---|---|---|---|---|---|
| base | 0.587 |  | 0.463 |  | 0.853 |  | 0.610 |  |
| SFT | 0.597 | +0.010 [-0.043, +0.063] | 0.532 | +0.069 [+0.044, +0.095] | 0.873 | +0.020 [-0.020, +0.060] | 0.597 | -0.013 [-0.070, +0.043] |
| DPO | 0.653 | +0.067 [+0.013, +0.120] | 0.556 | +0.093 [+0.068, +0.118] | 0.893 | +0.040 [-0.003, +0.083] | 0.623 | +0.013 [-0.043, +0.067] |
| SFT→DPO | 0.590 | +0.003 [-0.053, +0.060] | 0.580 | +0.117 [+0.089, +0.144] | 0.903 | +0.050 [+0.007, +0.093] | 0.637 | +0.027 [-0.023, +0.080] |
| GRPO (grader) | 0.613 | +0.027 [-0.027, +0.083] | 0.525 | +0.062 [+0.039, +0.086] | 0.903 | +0.050 [+0.010, +0.090] | 0.603 | -0.007 [-0.060, +0.043] |
| GRPO (RM) | 0.640 | +0.053 [-0.007, +0.110] | 0.572 | +0.109 [+0.084, +0.133] | 0.907 | +0.053 [+0.013, +0.097] | 0.613 | +0.003 [-0.050, +0.057] |
| SFT→GRPO | 0.633 | +0.047 [-0.007, +0.103] | 0.548 | +0.085 [+0.059, +0.112] | 0.867 | +0.013 [-0.033, +0.057] | 0.600 | -0.010 [-0.060, +0.040] |
| PPO (RM) | 0.487 | -0.100 [-0.157, -0.043] | 0.446 | -0.017 [-0.041, +0.008] | 0.837 | -0.017 [-0.060, +0.027] | 0.563 | -0.047 [-0.100, +0.007] |

- pass@1 vs base: every method except PPO is higher, intervals excluding zero,
  +0.062 to +0.117. SFT→DPO is again highest. Paired differences on SVAMP:
  SFT→DPO minus SFT→GRPO +0.031 [+0.006, +0.058]; SFT→DPO minus DPO +0.024
  [+0.000, +0.048]; SFT→GRPO minus GRPO +0.023 [-0.001, +0.048].
- PPO greedy is below base by 0.100 [-0.157, -0.043].
- Best-of-8 (RM) minus pass@1 is +0.042 to +0.147, smaller than on GSM8K test
  (+0.108 to +0.215); the GRPO (RM) interval includes zero (+0.042 [-0.003, +0.084]).
  Best-of-8 is 0.24 to 0.30 below pass@8.
- Answer format follows the GSM8K pattern: GRPO (RM) tags 12% of answers (strict
  pass@1 0.074 vs base 0.313), PPO tags 94% (`results/summary_svamp.md`).

#### Findings

- pass@1 vs base: all methods except PPO are higher on both splits, with
  intervals excluding zero. Range on test: +0.067 (DPO) to +0.140 (SFT→DPO).
- SFT initialisation: DPO and GRPO started from the SFT model have higher pass@1
  than the same methods started from base. Paired differences on test: DPO +0.074
  [+0.051, +0.097], GRPO +0.026 [+0.003, +0.049] (dev: +0.084, +0.042). Greedy
  differences: DPO -0.007 [-0.060, +0.047], GRPO +0.003 [-0.053, +0.060].
- SFT→DPO has the highest pass@1 on both splits. Paired difference to the next
  method, SFT→GRPO: test +0.043 [+0.021, +0.064], dev +0.049 [+0.027, +0.072].
  Greedy difference: test -0.013 [-0.070, +0.043].
- Greedy vs base: +0.023 to +0.047 on test, no interval excludes zero. Dev:
  +0.060 to +0.103. The greedy intervals are about ±0.05 wide, compared with about
  ±0.025 for pass@1, which averages 8 samples per question.
- pass@8 vs base on test: only SFT→DPO has an interval excluding zero, +0.063
  [+0.020, +0.107]. The others are -0.047 (PPO) to +0.040, intervals including zero.
- Best-of-8 (RM) vs base on test: -0.010 to +0.033 for every method except PPO,
  all intervals including zero; PPO is -0.073 [-0.130, -0.017]. On dev only the
  three SFT-initialised arms exclude zero (+0.053 to +0.057). Best-of-8 on the base
  model (test 0.583) is above the pass@1 of every trained model (at most 0.508).
- Best-of-8 (RM) minus pass@1: +0.108 (SFT→DPO) to +0.215 (base) on test,
  +0.140 to +0.253 on dev. Best-of-8 is 0.17 to 0.21 below pass@8 on test. On
  that share of questions at least one sample is correct and the reward model
  picks an incorrect one.
- PPO: no improvement over base. The first run (reward model without truncated-answer
  pairs, EOS penalty 1) reached KL ≈ 21; dev greedy accuracy was 0.137 and 90% of
  answers hit the token limit (base: 2%). `probe_rm.py` shows that this reward model
  scores truncated answers above complete ones. With `rm_trunc2_q` the KL was about
  3.5 and 0.5% of answers hit the limit. Dev: greedy 0.567, pass@1 0.435 (pass@1
  vs base -0.009 [-0.034, +0.016]). Test: pass@1 vs base -0.029 [-0.053, -0.005].

#### Answer format (test)

The prompt asks for a final line `Answer: <number>`. The grader takes the last
`Answer:` value, or the last number in the text if the tag is missing, so the
pass@1 above counts untagged answers. Each sampled answer is classed as
truncated (reaches 512 tokens), tagged, or untagged. Strict pass@1 counts only
complete, tagged, correct answers. Produced by `summarize.py`.

| method | truncated | tagged | untagged | acc. tagged | acc. untagged | pass@1 | strict pass@1 | Δ strict pass@1 vs base [95% CI] |
|---|---|---|---|---|---|---|---|---|
| base | 0.029 | 0.613 | 0.358 | 0.392 | 0.353 | 0.368 | 0.240 |  |
| SFT | 0.033 | 0.545 | 0.421 | 0.483 | 0.444 | 0.452 | 0.263 | +0.023 [+0.000, +0.046] |
| DPO | 0.057 | 0.372 | 0.571 | 0.461 | 0.460 | 0.435 | 0.172 | -0.068 [-0.093, -0.044] |
| SFT→DPO | 0.030 | 0.311 | 0.659 | 0.535 | 0.518 | 0.508 | 0.167 | -0.073 [-0.101, -0.047] |
| GRPO (grader) | 0.042 | 0.452 | 0.506 | 0.482 | 0.432 | 0.439 | 0.218 | -0.022 [-0.047, +0.001] |
| GRPO (RM) | 0.059 | 0.035 | 0.906 | 0.393 | 0.479 | 0.450 | 0.014 | -0.226 [-0.253, -0.200] |
| SFT→GRPO | 0.023 | 0.590 | 0.387 | 0.483 | 0.463 | 0.465 | 0.285 | +0.045 [+0.021, +0.069] |
| PPO (RM) | 0.007 | 0.926 | 0.067 | 0.351 | 0.199 | 0.339 | 0.325 | +0.085 [+0.061, +0.110] |

- Within each method except PPO and GRPO (RM), tagged and untagged answers
  differ in accuracy by 0.00 to 0.05, so the pass@1 gains are not explained by
  the fallback parser. GRPO (RM) tags too few answers (3.5%) to compare.
- DPO, SFT→DPO and GRPO (RM) tag fewer answers than base. GRPO (RM) tags 3.5%.
  Under strict parsing SFT→DPO (0.167) is below base (0.240), and the highest
  strict pass@1 is PPO (0.325), followed by SFT→GRPO (0.285).
- Strict pass@1 vs base on test, paired intervals: above base PPO +0.085
  [+0.061, +0.110], SFT→GRPO +0.045 [+0.021, +0.069], SFT +0.023 [+0.000, +0.046];
  below base DPO -0.068 [-0.093, -0.044], SFT→DPO -0.073 [-0.101, -0.047], GRPO (RM)
  -0.226 [-0.253, -0.200]; GRPO (grader) -0.022 [-0.047, +0.001].
- The grader reward used by GRPO and SFT→GRPO is `is_correct`, which accepts
  untagged answers. DPO and reward-model pairs are chosen by correctness only.
  None of these training signals rewards the tag directly.
- Dev shows the same pattern (`results/summary_dev.md`).

#### Wall-clock

Training time from the TRL logs (`train_runtime`; for PPO, first to last write of
the log). All runs were on one M3 Max laptop that also ran unrelated jobs, so
these are measured times, not controlled throughput. The three GRPO runs each
generated about 4,000 completions (500 prompts × 8). SFT trained on 1,507 correct
base-model answers for 2 epochs, DPO and SFT→DPO on 1,295 pairs for 1 epoch, PPO
for 500 episodes.

| arm | training (h) | dev eval (min) |
|---|---|---|
| SFT | 0.41 | 9.2 |
| DPO | 1.29 | 14.8 |
| SFT→DPO | 0.49 | 15.0 |
| GRPO (grader) | 3.89 | 16.0 |
| GRPO (RM) | 10.84 | 132.8 |
| SFT→GRPO | 2.55 | 12.0 |
| PPO (RM) | 0.47 | 6.2 |
| reward model v1 | 0.75 | |
| reward model `rm_trunc2` | 2.04 | |
| reward model `rm_trunc2_q` | 0.51 | |

The GRPO (RM) training and dev eval times are high relative to the other arms
and overlapped with other jobs on the machine; their share of contention was not
measured. DPO and SFT→DPO trained on the same number of pairs and differ by 2.6×
in time; contention is the likely cause but was not measured either.

</details>

### Merge bug (fixed)

`merge.py` used to merge every LoRA adapter onto the hub base model. For SFT→DPO
and SFT→GRPO the adapter was trained on top of `checkpoints/sft_lora/merged`, so
the merged model did not contain the SFT update (on one layer the merged weights
differed from base by 9.5e-5, while SFT differs from base by 8.5e-4). It now merges
onto the adapter's recorded `base_model_name_or_path`. The earlier SFT→DPO dev
result (greedy 0.617, pass@1 0.580) came from the wrong merge and is kept only as
`results/sftdpo_lr5e6_wrongbase_dev.json`.

## Limitations

- n=300 questions per split. Greedy differences below about 0.05 are not resolved.
  MDE not computed.
- One seed per method. Seed variance is not measured; dev-to-test changes in
  ranking cannot be attributed to the method or to the seed.
- 0.5B model, LoRA only, short runs that fit beside other jobs on one laptop
  (SFT→GRPO: 250 steps, 2 prompts × 8 samples, 300 tokens, GPU memory capped at 8 GB).
- All training data and reward-model pairs are sampled from the base model, one
  round only. The model never trains on its own improved answers.
- Settings were chosen on dev, so dev numbers are biased upward. Test was read once,
  after all settings were fixed.
- Intervals are per comparison, with no correction for the 7 comparisons against
  base per metric, or for the 5 metrics per comparison.
- The best-of-8 reward model `rm_trunc2` and the PPO reward `rm_trunc2_q` are
  trained on the same pairs, sampled from the base model.
- GRPO (RM) used reward model v1, which scores truncated answers above complete
  ones (`probe_rm.py`). GRPO (RM) has the highest truncation rate on test (0.059).

## Future work

Toward teaching the model more math:

- Several rounds of sample → grade → train (expert iteration), so later rounds
  learn from the improved model's answers rather than only from the base model's.
- Full rejection-sampling pipeline: best-of-8 with the reward model to build SFT data.
- Full fine-tuning instead of LoRA.

Toward a cleaner method comparison:

- More seeds per method, to separate method from seed.
- Longer PPO with KL snapshots along the run, to trace accuracy against KL.
- GRPO with `rm_trunc2` as reward.

## Reproducing

### Repository layout

    posttrain/
      common.py, generation.py   prompt, grader, file helpers; batched sampling with mlx-lm
      prep/      prepare.py (GSM8K splits), prepare_transfer.py (SVAMP), build_data.py (SFT and DPO data)
      train/     offline.py (SFT, DPO, reward model), grpo.py, ppo.py, merge.py (LoRA -> full checkpoint)
      eval/      evaluate.py, compare.py (paired bootstrap), bon_rm.py (best-of-8), probe_rm.py
      report/    summarize.py (tables and figures for this README)
    mlx/         alternative SFT, DPO and GRPO training with mlx-lm-lora (configs and reward)
    results/     evaluation summaries, summary tables and plots

Every script runs as a module from the repository root, e.g.
`.venv/bin/python -m posttrain.eval.evaluate --help`. `data/`, `checkpoints/`,
`logs/` and `adapters/` are created by the scripts and not committed.

### Setup

Two virtual environments, because PPO needs an older TRL:

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    python -m venv .venv-ppo && .venv-ppo/bin/pip install -r requirements-ppo.txt

### Data

    .venv/bin/python -m posttrain.prep.prepare      # GSM8K -> data/prompts/{train,valid,dev,test}.jsonl
    .venv/bin/python -m posttrain.prep.build_data   # 8 base-model samples per prompt, graded -> data/sft, data/dpo

### Training

Commands for the runs reported above (each output must be merged with
`merge.py` before it is used as `--init` or as a reward model):

    .venv/bin/python -m posttrain.train.offline --method sft --name sft_lora
    .venv/bin/python -m posttrain.train.offline --method dpo --name dpo_lr5e6 --lr 5e-6
    .venv/bin/python -m posttrain.train.offline --method dpo --name sftdpo_lr5e6 --lr 5e-6 --init checkpoints/sft_lora/merged
    .venv/bin/python -m posttrain.train.offline --method rm --name rm    # reward model v1
    .venv/bin/python -m posttrain.train.offline --method rm --name rm_trunc2 --trunc-neg --epochs 2
    .venv/bin/python -m posttrain.train.offline --method rm --name rm_trunc2_q --trunc-neg --epochs 0.5
    .venv/bin/python -m posttrain.train.grpo --name grpo
    .venv/bin/python -m posttrain.train.grpo --name rlhf_grpo --reward-model checkpoints/rm/merged
    .venv/bin/python -m posttrain.train.grpo --name sftgrpo --init checkpoints/sft_lora/merged --max-tokens 300 --grad-accum 4
    .venv-ppo/bin/python -m posttrain.train.ppo --reward-model checkpoints/rm_trunc2_q/merged --out checkpoints/ppo_rmtrunc_small \
        --episodes 500 --batch-size 4 --response-length 300 --missing-eos-penalty 1

Checkpoints are written to `checkpoints/<name>`. `--init <merged dir>` starts a
run from an earlier one.

`--trunc-neg` adds pairs in which a complete answer is preferred over the same
answer cut short, with and without an end-of-sequence token. Without them the
reward model scores unfinished answers above finished ones, and PPO learns to
stop emitting EOS.

`mlx/configs/*.yaml` and `mlx/rewards.py` are for training SFT, DPO and GRPO with
mlx-lm-lora instead of TRL. Those adapters are written to `adapters/` and are
evaluated with `evaluate.py --adapter`.

### Evaluation

    .venv/bin/python -m posttrain.train.merge checkpoints/sft_lora                # -> checkpoints/sft_lora/merged
    .venv/bin/python -m posttrain.train.merge checkpoints/rm_trunc --seq-cls      # reward model
    .venv/bin/python -m posttrain.eval.evaluate --name sft_lora_dev --model checkpoints/sft_lora/merged
    .venv/bin/python -m posttrain.eval.evaluate --name base_dev
    .venv/bin/python -m posttrain.eval.compare base_dev sft_lora_dev

`evaluate.py` generates one greedy answer and 8 sampled answers (temperature
0.8) per question and writes `results/<name>.json` and
`results/<name>.samples.jsonl`. Reported metrics: greedy accuracy, mean accuracy
of the sampled answers, pass@k (Chen et al. 2021), mean answer length in tokens,
and the fraction of answers that hit the token limit. `--split test` evaluates
on the test set.

`compare.py` computes the difference between two runs with a 95% bootstrap
interval. It resamples questions, not individual answers, and uses the same
resampled questions for both runs.

Reward-model diagnostics:

    .venv/bin/python -m posttrain.eval.probe_rm checkpoints/rm/merged checkpoints/rm_trunc2/merged
    .venv/bin/python -m posttrain.eval.bon_rm checkpoints/rm_trunc2/merged

`probe_rm.py` checks whether a reward model scores truncated answers above
complete ones. `bon_rm.py` measures the accuracy obtained by letting the reward
model choose one of the 8 stored samples per question (best-of-8).

Tables and plots:

    .venv/bin/python -m posttrain.eval.bon_rm checkpoints/rm_trunc2/merged --policy sft_lora --split dev
    .venv/bin/python -m posttrain.report.summarize --split dev     # -> results/summary_dev.md, results/plots/
    .venv/bin/python -m posttrain.report.summarize --figures       # README figures: gains, vs_sft, format_test

Transfer set:

    .venv/bin/python -m posttrain.prep.prepare_transfer          # SVAMP test -> data/prompts/svamp.jsonl
    .venv/bin/python -m posttrain.eval.evaluate --name sft_lora_svamp --model checkpoints/sft_lora/merged --split svamp
    .venv/bin/python -m posttrain.eval.bon_rm checkpoints/rm_trunc2/merged --policy sft_lora --split svamp
    .venv/bin/python -m posttrain.report.summarize --split svamp
