Below is the full randomness usage plan for every part of the pipeline where randomness can affect outputs (including places you haven’t implemented yet).

* **Global randomness contract**

  * **Control knob:** `sim_cfg["seed"]`

    * `null` ⇒ non-deterministic (fresh random realization each trial/run).
    * `int` ⇒ deterministic/reproducible.
  * **Recommended behavior when seed is set (assumption):** trials are still **different** but **reproducible**, by deriving a deterministic `trial_seed = f(base_seed, trial_idx)` (so trial 0/1/2 are fixed across reruns).
  * **Independence:** randomness should be split by **trial**, **group**, and **purpose** (inputs vs placement vs weights) so refactors don’t change unrelated randomness.

* **Step 2.3 Inputs generation (all spike-train randomness)**

  * **Homogeneous Poisson mode**

    * Randomness affects: spike times (exponential ISIs / Poisson events) for each synapse train.
    * Plan:

      * Per trial: regenerate trains (new RNG) if `seed=null`.
      * If seed set: use per-trial deterministic RNG (`trial_seed`) so each trial differs but is reproducible.
  * **Inhomogeneous Poisson mode (to implement)**

    * Randomness affects: spike times drawn from time-varying rate curve (thinning or bin-wise Poisson).
    * Plan:

      * Same as above, but also ensure baseline pre/post blocks (homogeneous) use the **same group input RNG stream** (so the whole train is consistent).
      * Baseline tokens like `"start"` are interpreted **inside the mode** (after loading the curve); timing layer stays numeric-only.
  * **Precomputed mode (to implement)**

    * Randomness affects only the **assignment** if `n_trains != N_syn` (e.g., sample trains with/without replacement vs deterministic tiling).
    * Plan:

      * Prefer deterministic mapping (truncate/tile) unless you explicitly want random sampling; if random sampling is used, it uses the group input RNG stream.
  * **Any “jitter” or event-time noise (future)**

    * If you add spike-time jitter, it must use the group input RNG stream.

* **Step 2.4 Synapse attachment (placement + weights randomness)**

  * **Synapse placement**

    * Randomness affects: which segments get synapses (sampling from eligible segments; potentially weighted by `dist_func`).
    * Plan:

      * Use a dedicated RNG stream per (trial, group, purpose=`placement`).
  * **Synaptic weights**

    * Randomness affects: lognormal weight draw (e.g., `wt_mean`, `wt_std`) per synapse.
    * Plan:

      * Use a separate RNG stream per (trial, group, purpose=`weights`) so placement refactors don’t change weights.
  * **Delays / conduction jitter (future)**

    * If you ever randomize synaptic delays, give it its own stream (purpose=`delays`) or fold into weights stream intentionally.

* **Step 2.5 Simulation dynamics**

  * If your NEURON mechanisms are purely deterministic HH + deterministic synapses (typical), then **no extra randomness** occurs during the run.
  * If you later add stochastic channels/synapses/noise-current injections, they need explicit RNG handling (preferably passed in or derived like the others).

* **Multi-trial logic (`run_multi`)**

  * To satisfy “different each trial unless seed is set”:

    * `seed=null` ⇒ call `generate_inputs(...)` and `add_synapses(...)` **inside the trial loop**, with unseeded RNGs (new realization each trial).
    * `seed=int` ⇒ still call them inside the trial loop, but use `trial_seed = f(base_seed, trial_idx)` so trials differ but are reproducible.
  * Important: if you generate inputs once outside the loop, inputs will be reused across trials regardless of `seed=null`.

* **Multi-sim / sweeps (future, including parametric later)**

  * Randomness affects: any Monte Carlo sampling of parameter sets, and of course inputs/placement/weights as above.
  * Plan:

    * Each run gets a run-id-derived seed stream (`run_seed = f(base_seed, run_id)`), and then trials derive from that if you also do trials.

* **Saving for QA/debugging**

  * Always record in outputs:

    * `base_seed` (may be null),
    * `trial_idx`,
    * `trial_seed_used` (null if base_seed null),
    * optionally per-group stream identifiers (not required, but helpful).
  * This is the sanity hook that lets you reproduce or explain “why did this trial look weird” later.

If you paste your current `run_multi` and where `generate_inputs` is called, I can tell you in one pass whether you currently regenerate inputs per trial and whether “seed set ⇒ reproducible but different trials” is already true or needs the `trial_seed` derivation.
