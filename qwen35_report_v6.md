# Qwen3.5-35B-A3B distributed GRPO on TPU v5p — reproduction report (v6)

Supersedes `qwen35_report_v4.md` / `_v5.md`. Written against
`gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-e2e-0912head-v8`
(Yixuan's e2e image) plus a 7-file overlay.

Everything here was measured on cluster `bodaborg-v5p-nap`, project
`cloud-tpu-shared-capacity`, region `europe-west4`, namespace `trellis`.

---

## 0. TL;DR for someone reproducing this

```bash
export KUBECONFIG=~/.kube/config.cloud-tpu-shared-capacity.europe-west4.bodaborg-v5p-nap
cd ~/git/tunix && ./build_qwen35_overlay.sh            # builds + pushes the overlay image
cd tunix/experimental/examples/math_gsm8k_dist
MAX_STEPS=2 RUN_TAG=smoke ./run_qwen35_repro.sh start  # smoke test first
./run_qwen35_repro.sh stop
```

Two things to know before you start:

1. **`$USER` is `igorts_google_com` on these VMs and underscores are illegal in
   k8s names.** `run_qwen35_repro.sh` sanitizes it (`cut -d_ -f1`). If you
   invoke the launcher directly you must do the same or every `kubectl apply`
   fails.
2. **`KUBECONFIG` is not the default path.** Without the export above, `kubectl`
   tries `localhost:8080` and refuses the connection.

---

## 1. What actually limits this run

This is the headline result and it is not what I expected going in.

Measured on a clean 2-step run (8-chip trainer, 32 rollout chips, packing on):

| Phase | Step 0 | Step 1 (steady state) |
|---|---|---|
| Weight sync | 258.25 s + 234.55 s | 233.39 s |
| Rollout + train | ~460 s (incl. 149 s compile) | ~160 s |
| **Total `step_time`** | **696.45 s** | **393.46 s** |

**Weight sync is 59% of a steady-state step.** Sequence packing cut a step from
32 microbatches to 5 and bought about 4 seconds, which means gradient
computation was never the bottleneck and never was. Of the ~160 s of non-sync
time, essentially all of it is rollout generation.

### The 100-steps-in-2-hours target

2 hours over 100 steps is a 72 s/step budget. Weight sync alone is 233 s. Even
with rollout and training at literally zero, 100 steps costs **6.5 hours**. The
target is not reachable by tuning batch shapes or by scaling the trainer for
throughput; it requires making weight sync itself cheaper, and probably
overlapping it.

Options, in descending order of expected payoff:

- **Overlap sync with rollout (`max_staleness > 0`).** Currently
  `max_staleness=0`, so generation waits for the sync to land. Allowing one
  step of staleness lets rollout run against the previous policy while the
  sync proceeds, hiding most of 233 s. This changes the algorithm from strictly
  on-policy GRPO to slightly off-policy — a legitimate and common choice, but a
  *semantic* change, not a tuning knob. Flagging rather than doing it.
- **Fewer rollout destinations.** Raiden fans the model out to every rollout
  worker; if the cost is per-destination, 8 workers is 8 copies. Memory on the
  rollout side is slack (see §5), so fewer, busier replicas may cost little
  generation throughput. Untested.
- ~~More trainer hosts.~~ **Measured, and it goes the wrong way — see §1.1.**

### 1.1 Scaling the trainer makes it *worse* (measured)

The intuition was that more trainer hosts would give Raiden's D2H more parallel
egress. It does not. Same run, same everything else, trainer slice swapped:

| Trainer | Chips | Hosts | FSDP | Weight sync |
|---|---|---|---|---|
| `tpuv5p:2x2x2` | 8 | 2 | 8 | **233–258 s** |
| `tpuv5p:2x2x4` | 16 | 4 | 16 | **345 s** |

Doubling the trainer made the dominant cost ~48% worse. More FSDP shards means
more, smaller per-shard transfers, and the per-shard overhead dominates any
gain in parallel egress.

This is convenient, because 8 chips is also the memory floor: full-parameter
Adam on 35B needs ~560 GB of params + grads + fp32 moments + master weights
against 95 GB × 8 = 760 GB. **8 trainer chips is both the minimum that fits and
the fastest for weight sync. Leave `TRAINER_TPU_SLICE` alone.**

Trainer chip count is therefore *not* the lever. Do not spend capacity there.

---

## 2. Image and overlay

Base: `yixuannwang_google_com-runner:yixuann-e2e-0912head-v8` — jax 0.11.0,
libtpu 0.0.44, vllm 0.28.1rc1 with the TPU backend, MaxText at `/app/maxtext`,
the Raiden FFI bridge, and a tunix checkout at `/app/tunix` that
`PYTHONPATH=/app` makes win over the pip-installed copy.

The overlay patches **7 files — 6 under `/app/tunix`, 1 under `/app/maxtext`**.
Keep it this small: the base image's tunix is a different lineage from
`github.com/google/tunix` HEAD, so copying the whole tree in would silently
downgrade Yixuan's fixes.

| # | File | Why |
|---|---|---|
| 1 | `maxtext/src/maxtext/training_engine/maxtext_engine.py` | MaxText PR #5219 — stop per-microbatch recompiles |
| 2 | `orchestrator/distributed_rl_engine.py` | round-robin rollout routing |
| 3 | `examples/math_gsm8k_dist/run_gsm8k_dist_grpo.py` | stop setting `prefix_hash` |
| 4 | `examples/math_gsm8k_dist/gsm8k.py` | log completions at INFO (diagnostic) |
| 5 | `examples/common/run_rollout_node.py` | `data_parallel_size` for the `vllm` sampler |
| 6 | `utils/maxtext_utils.py` | let `max_seq_token_per_tpu` raise `max_target_length` |
| 7 | `examples/common/run_trainer_node.py` | plumb `--max_seq_token_per_tpu` |

The Dockerfile asserts each patch is present at build time, so a moved upstream
path fails the build instead of the job.

### 2.1 The recompilation fix (MaxText PR #5219)

**Symptom:** the trainer recompiled the entire 35B graph on every microbatch.

**Cause:** `RLTrainerPayload` is a `flax.struct.dataclass` and `metadata` is
declared `pytree_node=False`, i.e. *static* aux data in the PyTreeDef. Its keys
change from microbatch to microbatch, so every microbatch presented a new
treedef and missed the jit cache.

`MaxTextTrainingEngine._prepare_batch` already excluded `"metadata"` on its
fallback dataclass→dict path, but forwarded the payload intact when a
`_gen_model_input_fn` was set — which is exactly our path
(`_algo_model_input`). PR #5219 clears it to `{}` via `dataclasses.replace`
before either branch.

Cherry-picked into `~/git/maxtext` as `96ccd858b` (`git cherry-pick -x
59420af4`). **Verified:** step 0 compiles once (149 s) then runs 31 × 2.45 s;
step 1 runs straight through.

> An earlier iteration of this work patched the equivalent fix into tunix's
> `orchestrator/algorithm_adapter.py`. That has been reverted in favour of
> #5219, which is merged upstream and sits one layer down. Do not reapply it.

### 2.2 A note on `639cbe7be`

The image's `maxtext_engine.py` is byte-identical to our local checkout *except*
for local commit `639cbe7be` (drain async checkpoint saves under Pathways).
Since the overlay ships the local file wholesale, that commit now rides along
with #5219. It is **inert here** — with `CHECKPOINT_SAVE_INTERVAL_STEPS=0` the
save path never fires — and it was kept deliberately: shipping a file that
matches no git commit is worse for reproducibility than shipping one extra
unreachable change.

---

## 3. Sequence packing

**Symptom:** packing appeared to be enabled but did nothing.

**Cause, and it is a nasty one:** `validate_packing_budget` requires
`max_seq_token_per_tpu >= max_prompt_length + max_response_length`. Stock
`maxtext_utils.build_maxtext_config` *pins* MaxText's `max_target_length` to
that same sum. So the budget is squeezed to exactly one maximal trajectory from
both sides at once, and a "packed" row holds exactly one sequence.

**Fix (overlay files 6 and 7):** add a `max_seq_token_per_tpu` kwarg to
`build_maxtext_config` and take `max(max_seq_token_per_tpu, max_prompt +
max_response)` for `max_target_length`, then plumb `--max_seq_token_per_tpu`
through `run_trainer_node.py`. (`k8s_launcher.sh` already passed the value to
the orchestrator — only the trainer was missing it.)

**Measured**, same step, 256 rollouts:

| `MAX_SEQ_TOKEN_PER_TPU` | trajectories/microbatch | microbatches/step |
|---|---|---|
| 1024 (the floor) | 8–16 (~1.14/row) | 32 |
| 4096 | 61, 55, 57, 48, 35 (~7.6/row) | **5** |

Worth doing, but see §1: it bought ~4 s of a 393 s step.

---

## 4. `train_micro_batch_size` propagation

A known failure mode, and worth stating explicitly because a teammate hit it:
`GRPOAdapter` used to default `train_micro_batch_size` to 1, and
`rl_program.py:206` reads it back via `getattr(algo, "train_micro_batch_size",
1)` — so the flag was validated and logged but never applied, and packing
collapsed to one sequence per pass.

**This is already fixed in the base image** (`run_gsm8k_dist_grpo.py:249`); it
is not something this overlay adds. Two confirmations on our runs:

- Trainer side, by construction: `maxtext_utils.py:161` raises if
  `train_micro_batch_size % mesh_fsdp != 0`. With `mesh_fsdp=8`, a leaked
  default of 1 gives `1 % 8 = 1` → hard startup failure. We start fine, so 8
  arrives. `per_device_batch_size = 8/8 = 1.0`.
- Orchestrator side, logged: `train_micro_batch_size=8`, `pack_size: 8`.

Note the asymmetry: the trainer *crashes* on a bad value, so a silently
sub-optimal step time from this bug comes from the orchestrator/packing half.

---

## 5. Topology, and why the architecture is left alone

**Trainer:** Pathways, `tpuv5p:2x2x2` = 8 chips, FSDP=8 / TP=1 / EP=1.
v5p packs 4 chips per host, so this is 2 hosts.

**Rollout:** 8 × `tpuv5p:2x2x1` = 4 chips each, 32 total, each filled as
**TP=2 × DP=2**.

TP must stay at 2, and this is the one place where a tuning knob would have
quietly changed the science:

- At **TP=4**, `maxtext_utils` replicates `base_num_kv_heads` 2 → 4, and GMM_v2
  pads the MoE MLP dim 512 → 1024 (512/4 is not a multiple of 2·128). The MoE
  experts are ~32B of this 35B model, so that padding roughly doubles what the
  trainer holds — and it means training a *different model*.
- At **TP=2**, 512/2 = 256 is already aligned and `num_kv_heads` is untouched.
  Verified in-run: `kv_heads=2`, `moe_mlp_tp_size=2`,
  `padded_base_moe_mlp_dim=512`.

`base_num_kv_heads: 2` is therefore what pins the rollout's atomic unit at 2
chips; the only free variable on that side is replica count.

A sizing note: "A3B" (3B active) cuts compute, not optimizer state — all 35B
params carry it. Full-parameter Adam needs roughly 70 GB bf16 params + 70 GB
grads + fp32 moments + fp32 master ≈ 560 GB plus activations, against 95 GB ×
8 = 760 GB. 8 chips is the working floor, not a comfortable choice.

### `inprocess_vllm`, not `vllm`

**This is mandatory, and the failure is ugly.** On the plain `vllm` path the
Raiden destination registry is never populated with MaxText-named variables, so
weight sync dies in preflight before step 0:

```
WeightSyncError: round 0 ... manifest preflight failed before any destination
was quiesced (1266 problems, first: preflight: source variable
'decoder.decoder_norm.scale' has no destination counterpart)
```

1266 = 2 × 633 variables — *nothing* matched. `inprocess_vllm` logs `Using local
registration for destination metadata` and matches all 633. Root-caused by
controlled experiment: same image, same mesh, same everything else, sampler
flipped. `prefuse_moe_weights` on the trainer was investigated first and
produced a byte-identical failure, so it is not the cause (it is still set, to
match the rollout).

---

## 6. Open issues

### 6.1 BLOCKER: reward is pinned at exactly 0

`reward_mean = 0.0000`, `std = 0.0000` on every step of every run so far
(including the older v4 runs, so this predates all changes described here).
With `BETA=0` and GRPO advantage `(r − mean)/std`, zero reward fully explains
`loss = 0.0000` and `perplexity = 1.0000`: **there is no learning signal at
all, and a 100-step run in this state would be worthless.**

Two false starts are worth recording, because both were reasonable and both
were wrong, and the second one wasted a run.

**False start 1 — "the completions are truncated."** I read the zero-variance
reward as a truncation signature and raised `MAX_RESPONSE_LENGTH` 512 → 1024.
Per-completion logging (overlay file 4) disproved it: the logged completions
were 10–58 characters and ended by emitting `<|endoftext|>` / `<|im_end|>` on
their own, nowhere near the cap.

**False start 2 — "the model emits garbage, so weight sync must be broken."**
That is what those 10–58 character completions look like, and with vLLM started
on `load_format: 'dummy'` (random weights, installed for real only by Raiden
sync) a bad sync is a natural suspect. But **the sample was not
representative**, and this is the actual finding:

> Overlay file 4 logs from `GSM8KEnv._step_impl` — i.e. from inside
> `env.step`. `trajectory_collect_engine._one_step` only calls `env.step` when
> `_check_and_set_context_limit_reached()` is False. **A trajectory that spends
> its entire response budget in one turn never reaches `env.step` at all**, and
> `collect()` additionally skips `_append_final_reward()` for it. So it scores a
> hard 0.0 regardless of content, *and* it is invisible to the example's own
> completion logging.

On rollout worker 0 of the 512-budget run, **27 of ~30 trajectories were
clipped that way** (`trajectory clipped: MAX_CONTEXT_LIMIT_REACHED`, 27
occurrences). The vLLM progress lines confirm they were real, full-length
generations — 47.9 s at 10.7 tok/s ≈ 512 tokens, exactly the budget. The only
completions I had actually been looking at were the *rare* ones where the model
emitted EOS after ~1 token, which is why they all looked degenerate and
near-identical.

So the mechanism behind `mean = 0.0000, std = 0.0000` is now established and is
mundane: **~84% of trajectories were dropped before grading**, and a dropped
trajectory contributes a hard zero. Zero variance does not require a collapsed
model; it only requires that almost nothing gets graded.

Correspondingly, the earlier claim that "all 8 workers emit byte-identical
completions, so the distribution has collapsed" should be **withdrawn on both
counts**: it rested on 1-token `<|im_end|>` samples, and in any case vLLM is
configured with `seed=0`, so identical output across workers is expected
determinism, not evidence of anything.

#### What the raw generations actually show (image v11, `RUN_TAG=gendiag`)

Overlay file 8 logs generations from `_one_step`, before the overlong filter can
drop them. With `TUNIX_LOG_GENERATIONS=8` and a 1024 budget, every logged
trajectory used the **entire** budget — `tokens=1024`, ~1100 chars — and the
content is:

```
head='\nS\n*    1\n2 1 1: 6 6 6 0  \n\n  *   . \n\n,\n\n\n\n \n\n 8 0 3 6  0 a ...'
first_ids=[198, 50, 198, 9, 256, 256, 16, 198, 17, 220, 16, 220, 16, 25, 220, 21]
```

**The model is genuinely broken.** Two things sharpen this beyond "it's garbage":

- The token IDs are overwhelmingly `198` (`\n`), `220` (space), `16`/`17`/… —
  i.e. the *unigram frequency* distribution, plus digits, which are contextually
  frequent for a math prompt. That is the signature of a network whose
  embedding and unembedding are intact while the 40 transformer layers
  contribute noise: the residual stream carries no information, so the logits
  fall back to token frequency. It is *not* the uniform-random-unicode
  signature of fully random weights.
- The prompt is correct. The logged `chat_completions` is the proper VTC
  template — "put your detailed step-by-step reasoning process inside
  `<reasoning>`…", the GSM8K problem, and the deliberately-open `<reasoning>\n`.
  **So the `chat_parser=auto` hypothesis is dead**; a prompt problem cannot
  produce digit soup.

#### Weight sync transports the right values — and is still the problem

With `VERIFY_WEIGHTS=true` (which only works from v11 on, see below), source and
destination checksums are the same to within float accumulation order:

| | trainer (source) | rollout (destination) |
|---|---|---|
| `decoder_norm.scale` | 3232.55078125 | 3232.55078125 |
| `layers_0.attention.A_log` | 105.5625 | 105.5625 |
| `layers_0.attention.conv1d.kernel` | 561.030517578125 | 561.030517578125 |
| `__grand_total__` | 304540019.06 | 304540018.28 |
| `__tensor_count__` | 633 | 633 |
| `__element_count__` | 34,660,610,688 | 34,660,610,688 |

So: all 633 tensors, all 34.66B elements, right values, and the transport writes
in place into the live serving state (`RaidenWeightSyncDelegate` docstring:
*"the transport binds the live transformer state directly, so weight_sync writes
into the serving copy rather than a shadow buffer"*).

**The key inference:** these checksums are per-tensor *absolute sums*, which are
invariant under permutation and reshape. Right values + broken model therefore
points squarely at **the arrangement, not the transport** — the elements arrive
but land in the wrong positions.

Ruled out:

- **Not missing tensors.** `raiden bind dropped 2 unbindable leaves` names only
  `routed_experts.rngs.params.count` and `.key` — RNG state, correctly excluded.
- **Not the base checkpoint.** The trainer restores it cleanly
  (`[sync] Finished load in 95.96 seconds`), and its own source checksums look
  like trained weights, not init (e.g. `decoder_norm.scale` mean ≈ 0.79, not 1.0).
- **Not response length, not the prompt, not the chat parser**, per above.
- **Not caused by anything in this overlay** — v4 showed the same zero reward.

Remaining hypotheses, now narrowed to layout:

1. **MoE interleave.** `_interleave_moe_weights` (`tunix/generate/utils.py:1229`)
   is a pure reshape/permute parameterised by `n_shards` and `lane_size` — the
   exact class of transform that preserves an abs-sum while destroying the
   model. The MoE experts are ~32B of this 35B model. **Being tested now**
   (`RUN_TAG=noprefuse`, `PREFUSE_MOE_WEIGHTS=false` on both sides).
2. **Cross-mesh shard mapping.** The trainer is FSDP=8 over 2 hosts; each rollout
   is `data:2 × model:2` with `num_shards=4`. Raiden pairs tensors by name and
   ships shards; if the 8-way source sharding is mapped onto the 4-way
   destination incorrectly, every device gets the wrong slice and the global
   abs-sum is still exactly preserved. This fits the evidence just as well as
   (1) and is *not* excluded by the `noprefuse` test.
3. **Hybrid Gated-DeltaNet tensors.** 30 of 40 layers are GDN (`A_log`,
   `dt_bias`, `conv1d.kernel`, `in_proj_qkvz`, `in_proj_ba`) — unusual tensors
   with packed layouts, and a wrong unpacking of `in_proj_qkvz` would corrupt
   three-quarters of the network exactly as observed.

Note (1) and (2) are both *layout* faults and both consistent with everything
measured; the checksum test cannot separate them. Separating them needs
element-order verification, not sum verification — e.g. a per-tensor hash of the
first N elements in canonical order, compared across the two sides. That is the
diagnostic I would add next, and it is a small change to
`RaidenSynchronizer.checksums()`.

Two practical notes for whoever picks this up:

- `VERIFY_WEIGHTS` did not reach the container before v11:
  `run_qwen35_repro.sh` hard-coded `export VERIFY_WEIGHTS=false`, silently
  overriding the caller. One diagnostic run was lost to this. It is now
  `${VERIFY_WEIGHTS:-false}`. Confirm any diagnostic flag actually landed with
  `kubectl get pod ... -o jsonpath='{.spec.containers[0].command}'`.
- `DEBUG=1` is *not* a usable alternative at this scale — it routes
  httpx/httpcore through the node loggers and buries the run in HTTP headers.
  That is why overlay files 4 and 8 log at INFO with per-worker budgets
  (`GSM8K_LOG_COMPLETIONS`, `TUNIX_LOG_GENERATIONS`).

### 6.2 Trajectories that use the whole response budget are never graded

Worth stating separately from 6.1, because it is a real defect that will bite
any long-reasoning model even once the weights are fixed. In
`trajectory_collect_engine`:

- `_one_step` calls `env.step` only `if not
  _check_and_set_context_limit_reached()`, and
- `collect()` skips `_append_final_reward()` when the status is in
  `filter_statuses` and `overlong_filter` is on (it defaults to **True** in
  `TrajectoryCollectEngine.collect_batch`, line 347).

So a trajectory that spends its full budget scores a hard 0.0 rather than being
graded on what it produced. Under the VTC recipe that is the difference between
0.0 and the 0.1 a correctly-formatted-but-wrong answer earns — i.e. it removes
exactly the gradient signal GRPO needs. At `MAX_RESPONSE_LENGTH=512` this hit
27 of ~30 trajectories on one worker; at 1024 it still hit every logged one.

Raising the budget only postpones this. The real options are to grade clipped
trajectories on their partial output, or to leave `overlong_filter` on but
accept that the budget must exceed what the model actually emits. Decide this
deliberately — it is a change to what the reward means, so I have not made it.

### 6.3 Resolved this session

**JobSet restart-into-deadlock — fixed.** When the orchestrator died,
`RestartJobSetFailurePolicyAction` + `BackoffLimitExceeded` recreated the pod,
but the rollout workers stayed registered with the *dead* discovery server. The
new orchestrator then waited forever while 40 v5p chips sat idle and every pod
read `1/1 Running`. Fixed by parameterising `maxRestarts` in
`jobset.cpu.yaml` and defaulting the orchestrator to 0 (workers stay at 3), via
`ORCHESTRATOR_MAX_RESTARTS`. Confirmed working: the `gendiag` run's orchestrator
died and stayed `0/1 Error` instead of deadlocking the cluster.
**Monitor the `Program End: EXIT_CODE=` line,
not pod status.**

**Benign but misleading:** `No checkpoint found, skipping restore.` is the tunix
RL-resume probe against the empty run directory. Base weights load fine —
look for `[sync] Finished load in 95.96 seconds @ gs://hengtaoguo-maxtext-logs/...`.

---

## 7. Host-side plumbing (no image change)

- `yaml_generator.py`: added `--priority_class`, emitting the
  `kueue.x-k8s.io/priority-class` label. Without it the workload sits at
  priority 0 and anything in `tpu-shared-cohort` evicts it — an earlier 100-step
  attempt died at step 43 that way. `medium` = 500, applied to all 11 workloads.
- `k8s_launcher.sh`: exports `KUEUE_PRIORITY_CLASS`, passes `--priority_class`
  at 3 sites, and passes `--prefuse_moe_weights` / `--max_seq_token_per_tpu` to
  the trainer block.
- `PATHWAYS_PROXY_MEMORY_LIMIT=250G`: the whole model is staged on the proxy
  host during Raiden D2H; the generator's 100G default OOM-kills it at 35B.

## 8. Metrics

`WANDB_API_KEY` is read out of `~/.bashrc` by the launcher
(`eval "$(grep -m1 '^export WANDB_API_KEY=' ~/.bashrc)"`) because `.bashrc`
returns early for non-interactive shells, so sourcing it is not enough. The key
value is deliberately never printed or written to any report or manifest.
Project `qwen35-35b-a3b-grpo`; `FLUSH_METRICS_EVERY_N_STEPS=1`.
