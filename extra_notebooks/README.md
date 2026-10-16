# Extra Notebooks

Optional SCP notebooks that support specialized workflows but are not part of the
main numbered pipeline.

- [custom_cells.ipynb](custom_cells.ipynb): executable non-biological examples
  of live adoption, native-source copies, and CLI reconstruction. Requires the
  normal SCP environment; uses temporary workspaces without ACT/BMTool.

- `act_segmentation.ipynb`: cleaned SCP channel-segmentation helper for ACT-style
  workflows. Use it when you need to manually segment channel activation
  functions before passive/active tuning. It is not required for normal Step 1-7
  usage.

Extra notebooks should stay small, self-contained, and clearly optional. If a
workflow becomes required for the main public path, move the backend into
`modules/` and expose it from one of the numbered notebooks.
