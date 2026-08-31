# Flat 5.6 Hz PN test

`pn_exc_flat_5p6_test.json` is an archived replacement for the ordinary
`pn_exc.json` group. It deliberately keeps the group name `pn_exc`, so use only
one of those files at a time; do not list both in `syn_config.json`.

To activate the longer flat-input test:

1. In `../syn_config.json`, replace `syn_groups/pn_exc.json` with
   `syn_groups/pn_exc_flat_5p6_test.json`.
2. In `../sim_config.json`, set `tstop` to `1500.0` and
   `stim_duration_ms` to `1000.0`. The analysis stimulus is then 300–1300 ms.
3. Extend the `stop_ms` of the `bg_exc` and `bg_inh` background blocks to
   `1500.0`.
4. Extend the `stop_ms` of the `vip_inh` post-baseline block to `1500.0`.
5. Re-run the Step 5 setup/options cell so its widgets reload the changed
   simulation timing before launching the run.

To return to the ordinary SST AMPA configuration, reverse those changes:
reference `pn_exc.json`, use `tstop = 1000.0` and
`stim_duration_ms = 500.0`, and restore the other groups' final `stop_ms` to
`1000.0`.
