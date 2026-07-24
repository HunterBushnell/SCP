from __future__ import annotations

import ast
import inspect
import json
import unittest
from pathlib import Path
from typing import Iterable

from modules.notebooks.bootstrap import finish_step5_notebook_setup


REPO_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_NOTEBOOKS = (
    "0_pipeline.ipynb",
    "1_setup.ipynb",
    "2_passive.ipynb",
    "3_active.ipynb",
    "4_synapses.ipynb",
    "5_simulate.ipynb",
    "6_analysis.ipynb",
    "7_tools.ipynb",
    "extra_notebooks/act_segmentation.ipynb",
)

FORBIDDEN_RELEASE_CONTENT = (
    "/home/",
    "/Users/",
    "C:\\Users\\",
    "Traceback (most recent call last)",
    "KeyboardInterrupt",
)


def _code_cells(notebook_name: str) -> list[str]:
    notebook_path = REPO_ROOT / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def _parsed_cells(notebook_name: str) -> Iterable[tuple[int, str, ast.Module]]:
    for cell_index, source in enumerate(_code_cells(notebook_name)):
        try:
            tree = ast.parse(source, filename=f"{notebook_name}:cell-{cell_index}")
        except SyntaxError:
            # A notebook may legitimately contain IPython-only syntax. The
            # protocol/setup cells covered here are ordinary Python.
            continue
        yield cell_index, source, tree


def _assigned_value(notebook_name: str, variable_name: str) -> tuple[str, ast.AST]:
    for _, source, tree in _parsed_cells(notebook_name):
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(
                isinstance(target, ast.Name) and target.id == variable_name
                for target in targets
            ):
                continue
            return source, node.value
    raise AssertionError(f"No assignment to {variable_name!r} found in {notebook_name}")


def _assigned_dict(notebook_name: str, variable_name: str) -> tuple[str, ast.Dict]:
    source, value = _assigned_value(notebook_name, variable_name)
    if not isinstance(value, ast.Dict):
        raise AssertionError(f"{variable_name} must be assigned a dict literal")
    return source, value


def _assigned_literal(notebook_name: str, variable_name: str) -> object:
    _, value = _assigned_value(notebook_name, variable_name)
    return ast.literal_eval(value)


def _dict_value(mapping: ast.Dict, key_name: str) -> ast.AST:
    for key, value in zip(mapping.keys, mapping.values):
        if isinstance(key, ast.Constant) and key.value == key_name:
            return value
    raise AssertionError(f"Dict literal does not define {key_name!r}")


def _assert_reads_sim_conditions(test: unittest.TestCase, value: ast.AST) -> None:
    matching_gets = []
    for node in ast.walk(value):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "get" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "sim_config" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and node.args[0].value == "conditions":
            matching_gets.append(node)
    test.assertEqual(
        len(matching_gets),
        1,
        "runtime conditions must be copied from sim_config.get('conditions', ...)",
    )


def _calls(notebook_name: str, function_name: str) -> list[tuple[int, ast.Call]]:
    found: list[tuple[int, ast.Call]] = []
    for cell_index, _, tree in _parsed_cells(notebook_name):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called_name = node.func.attr
            else:
                continue
            if called_name == function_name:
                found.append((cell_index, node))
    return found


def _keyword(call: ast.Call, name: str) -> ast.AST:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"Call to {ast.unparse(call.func)} lacks keyword {name!r}")


class NotebookContractTests(unittest.TestCase):
    def test_public_notebooks_are_saved_clean_and_portable(self) -> None:
        for notebook_name in PUBLIC_NOTEBOOKS:
            with self.subTest(notebook_name=notebook_name):
                notebook = json.loads(
                    (REPO_ROOT / notebook_name).read_text(encoding="utf-8")
                )
                kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
                self.assertEqual(kernelspec.get("display_name"), "scp-py311")
                self.assertEqual(kernelspec.get("name"), "scp-py311")

                serialized = json.dumps(notebook, ensure_ascii=False)
                for forbidden in FORBIDDEN_RELEASE_CONTENT:
                    self.assertNotIn(forbidden, serialized)

                code_cells = [
                    cell
                    for cell in notebook.get("cells", [])
                    if cell.get("cell_type") == "code"
                ]
                self.assertTrue(code_cells)
                for cell in code_cells:
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs"), [])

    def test_compact_pipeline_has_hidden_bootstrap(self) -> None:
        notebook = json.loads(
            (REPO_ROOT / "0_pipeline.ipynb").read_text(encoding="utf-8")
        )
        code_cells = [
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ]
        self.assertTrue(code_cells)
        for cell in code_cells:
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])

        bootstrap = code_cells[0]
        self.assertTrue(
            bootstrap.get("metadata", {})
            .get("jupyter", {})
            .get("source_hidden")
        )

    def test_compact_pipeline_uses_one_controller_and_five_step_panels(self) -> None:
        self.assertEqual(len(_calls("0_pipeline.ipynb", "PipelineNotebookUI")), 1)
        for panel_method in (
            "step1_panel",
            "step2_panel",
            "step3_panel",
            "step4_panel",
            "step5_panel",
        ):
            with self.subTest(panel_method=panel_method):
                self.assertEqual(len(_calls("0_pipeline.ipynb", panel_method)), 1)

        for direct_call in (
            "prepare_pipeline_notebook",
            "run_passive_stage",
            "run_active_stage",
            "prepare_interactive_synapse_tuner",
            "preview_pipeline_inputs",
            "run_fresh_simulation",
            "show_run_diagnostics",
            "show_synapse_preview",
            "SingleEvent",
            "InteractiveTuner",
            "stp_frequency_response",
            "optimize_parameters",
        ):
            with self.subTest(direct_call=direct_call):
                self.assertEqual(_calls("0_pipeline.ipynb", direct_call), [])

        _, settings = _assigned_dict("0_pipeline.ipynb", "pipeline_settings")
        expected_literals = {
            "cell_name": "PV",
            "tune_name": "tuned",
            "quiet_step1_output": True,
            "passive_amps_pA": [-50, -100],
            "passive_target_overrides": {},
            "active_amps_pA": [150, 300],
            "active_spike_threshold_mV": -20,
            "active_include_currents": True,
            "active_current_display_amp_pA": None,
            "fi_protocol_overrides": {},
            "fi_spike_threshold_mV": -20,
            "act_active_module": None,
            "act_n_cpus": None,
            "act_workspace_override": None,
            "act_overrides": {},
            "act_overwrite_outputs": False,
            "enable_synapse_tuning": False,
            "n_trials": 1,
            "run_iclamp": False,
            "quiet_input_preview_output": True,
            "quiet_simulation_output": True,
            "simulation_overrides": {},
            "input_preview_groups": None,
            "input_preview_plots": [
                "weight_distribution",
                "distance_distribution",
                "weight_vs_distance",
            ],
            "input_preview_trial_idx": 0,
            "input_preview_show_table": True,
            "input_preview_histogram_density": True,
            "input_preview_distance_bin_um": 25.0,
            "input_preview_weight_bin": None,
            "input_preview_plot_columns": 3,
            "input_preview_plot_size": "compact",
            "diagnostic_plots": [
                "input_rate",
                "membrane_voltage",
                "output_rate",
                "output_raster",
            ],
            "diagnostic_trial_idx": 0,
            "diagnostic_window_mode": "stimulus",
            "diagnostic_window_start_ms": None,
            "diagnostic_window_stop_ms": None,
            "diagnostic_window_padding_ms": 100.0,
            "diagnostic_rate_bin_ms": None,
            "diagnostic_smoothing_ms": None,
            "diagnostic_raster_style": "dot",
            "diagnostic_input_groups": None,
            "diagnostic_show_stimulus": True,
            "diagnostic_figure_size": "compact",
        }
        for key, expected in expected_literals.items():
            with self.subTest(setting=key):
                self.assertEqual(ast.literal_eval(_dict_value(settings, key)), expected)
        setting_names = {ast.literal_eval(key) for key in settings.keys}
        self.assertNotIn("adb_specimen_id", setting_names)
        self.assertNotIn("adb_model_type", setting_names)
        self.assertNotIn("compute_act_passive_proposal", setting_names)
        self.assertEqual(
            ast.unparse(_dict_value(settings, "fi_amps_pA")),
            "list(range(0, 301, 50))",
        )

        notebook = json.loads(
            (REPO_ROOT / "0_pipeline.ipynb").read_text(encoding="utf-8")
        )
        all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("Run All does not start a model or simulation", all_source)
        self.assertIn("Run selected module", all_source)
        self.assertIn("Cancel", all_source)
        self.assertIn("Review evaluation", all_source)
        self.assertNotIn("widgets.Accordion", all_source)
        self.assertNotIn("widgets.Tab", all_source)

    def test_numbered_notebooks_use_pv_release_defaults(self) -> None:
        expected_cell_tunes = {
            "1_setup.ipynb": ("PV", "orig"),
            "2_passive.ipynb": ("PV", "tuned"),
            "3_active.ipynb": ("PV", "tuned"),
            "4_synapses.ipynb": ("PV", "tuned"),
            "5_simulate.ipynb": ("PV", "tuned"),
        }
        for notebook_name, (expected_cell, expected_tune) in expected_cell_tunes.items():
            with self.subTest(notebook_name=notebook_name):
                self.assertEqual(
                    _assigned_literal(notebook_name, "cell_name"),
                    expected_cell,
                )
                self.assertEqual(
                    _assigned_literal(notebook_name, "tune_name"),
                    expected_tune,
                )

        self.assertEqual(_assigned_literal("6_analysis.ipynb", "cell_name"), "PV")
        self.assertEqual(_assigned_literal("6_analysis.ipynb", "model_dir"), "tuned")

        tools_source = "\n".join(_code_cells("7_tools.ipynb"))
        self.assertIn("cells/PV/tunes/tuned", tools_source)
        self.assertNotIn("cells/SST/tunes/tuned", tools_source)

    def test_detailed_protocol_notebooks_use_pv_release_protocols(self) -> None:
        self.assertFalse(
            _assigned_literal("2_passive.ipynb", "COMPUTE_ACT_PASSIVE_PROPOSAL")
        )
        _, passive_params = _assigned_dict("2_passive.ipynb", "sim_params")
        self.assertEqual(ast.literal_eval(_dict_value(passive_params, "stim_delay")), 300)
        self.assertEqual(ast.literal_eval(_dict_value(passive_params, "stim_dur")), 1000)
        self.assertEqual(ast.literal_eval(_dict_value(passive_params, "h_tstop")), 1500.0)
        self.assertEqual(
            _assigned_literal("2_passive.ipynb", "sim_amps"),
            [-50, -100],
        )

        _, active_params = _assigned_dict("3_active.ipynb", "active_sim_params")
        _, fi_params = _assigned_dict("3_active.ipynb", "fi_sim_params")
        for params in (active_params, fi_params):
            self.assertEqual(ast.literal_eval(_dict_value(params, "stim_delay")), 200)
            self.assertEqual(ast.literal_eval(_dict_value(params, "stim_dur")), 1000)
            self.assertEqual(ast.literal_eval(_dict_value(params, "h_tstop")), 1500)
        self.assertEqual(
            _assigned_literal("3_active.ipynb", "active_sim_amps"),
            [150, 300],
        )
        self.assertEqual(
            _assigned_literal("3_active.ipynb", "FI_AMP_RANGE"),
            (0, 300, 50),
        )
        self.assertIsNone(_assigned_literal("3_active.ipynb", "PLOT_XLIM"))
        self.assertIsNone(_assigned_literal("3_active.ipynb", "CURRENT_YLIM"))

    def test_step1_synapse_scaffolding_is_allen_only_by_default(self) -> None:
        _, value = _assigned_value("1_setup.ipynb", "DO_SETUP_SYNAPSE_CONFIGS")
        self.assertIsInstance(value, ast.Compare)
        self.assertIsInstance(value.left, ast.Name)
        self.assertEqual(value.left.id, "cell_loader")
        self.assertEqual(len(value.ops), 1)
        self.assertIsInstance(value.ops[0], ast.Eq)
        self.assertEqual(len(value.comparators), 1)
        self.assertIsInstance(value.comparators[0], ast.Constant)
        self.assertEqual(value.comparators[0].value, "allen_manifest")

    def test_step2_and_step3_construct_one_cell_per_kernel(self) -> None:
        for notebook_name in ("2_passive.ipynb", "3_active.ipynb"):
            with self.subTest(notebook_name=notebook_name):
                build_calls = _calls(notebook_name, "build_tuning_cell")
                self.assertEqual(
                    len(build_calls),
                    1,
                    "tuning notebooks must reuse the cell built in their Build Cell "
                    "section because legacy Allen models cannot be constructed twice "
                    "in one NEURON process",
                )

    def test_step2_passive_protocol_copies_runtime_conditions(self) -> None:
        _, sim_params = _assigned_dict("2_passive.ipynb", "sim_params")
        _assert_reads_sim_conditions(self, _dict_value(sim_params, "conditions"))

        calls = _calls("2_passive.ipynb", "run_passive_protocol")
        self.assertEqual(len(calls), 1)
        passed_params = _keyword(calls[0][1], "sim_params")
        self.assertIsInstance(passed_params, ast.Name)
        self.assertEqual(passed_params.id, "sim_params")

        setup_source = "\n".join(_code_cells("2_passive.ipynb"))
        self.assertIn("COMPUTE_ACT_PASSIVE_PROPOSAL = False", setup_source)
        target_calls = _calls("2_passive.ipynb", "resolve_passive_tuning_inputs")
        self.assertEqual(len(target_calls), 1)
        proposal_flag = _keyword(target_calls[0][1], "compute_act_proposal")
        self.assertIsInstance(proposal_flag, ast.Name)
        self.assertEqual(proposal_flag.id, "COMPUTE_ACT_PASSIVE_PROPOSAL")

    def test_step3_active_and_fi_protocols_copy_runtime_conditions(self) -> None:
        for variable_name in ("active_sim_params", "fi_sim_params"):
            with self.subTest(variable_name=variable_name):
                _, params = _assigned_dict("3_active.ipynb", variable_name)
                _assert_reads_sim_conditions(self, _dict_value(params, "conditions"))

        calls = _calls("3_active.ipynb", "run_active_protocol")
        passed_names = []
        for _, call in calls:
            value = _keyword(call, "sim_params")
            if isinstance(value, ast.Name):
                passed_names.append(value.id)
        self.assertEqual(
            sorted(passed_names),
            ["active_sim_params", "fi_sim_params"],
        )

    def test_step1_passes_prospective_cell_config_to_mechanism_setup(self) -> None:
        calls = _calls("1_setup.ipynb", "prepare_mechanisms")
        self.assertEqual(len(calls), 1)
        configured_cell = _keyword(calls[0][1], "cell_config")
        self.assertIsInstance(configured_cell, ast.Name)
        self.assertEqual(configured_cell.id, "mechanism_cell_config")

        _, config_mapping = _assigned_dict("1_setup.ipynb", "mechanism_cell_config")
        self.assertIsInstance(_dict_value(config_mapping, "paths"), ast.Call)

        setup_source = "\n".join(_code_cells("1_setup.ipynb"))
        self.assertIn("MODEL_SOURCE_OVERRIDES = None", setup_source)
        self.assertEqual(len(_calls("1_setup.ipynb", "resolve_step1_model_setup")), 1)
        for removed_cell_specific_setting in (
            "HOC_TEMPLATE_FILE",
            "HOC_TEMPLATE_NAME",
            "HOC_CONSTRUCTOR_ARGS",
            "HOC_SECTION_MAP",
            "soma_diam_multiplier",
            "cell_color",
            "v_init_mV",
            "celsius_C",
        ):
            self.assertNotIn(removed_cell_specific_setting, setup_source)

        base_calls = _calls("1_setup.ipynb", "prepare_base_configs")
        self.assertEqual(len(base_calls), 1)
        passed_keywords = {keyword.arg for keyword in base_calls[0][1].keywords}
        self.assertTrue(
            {"soma_diam_multiplier", "color", "v_init_mV", "celsius_C"}.isdisjoint(
                passed_keywords
            )
        )

    def test_step1_generates_generic_mode_specific_target_templates(self) -> None:
        setup_source = "\n".join(_code_cells("1_setup.ipynb"))
        self.assertIn('TARGET_SOURCE_MODE = "manual"', setup_source)
        self.assertIn("INCLUDE_MANUAL_WITH_FILE_SOURCE = False", setup_source)
        for old_notebook_value in (
            "MANUAL_PASSIVE_TARGETS",
            "MANUAL_FI_CURRENTS_PA",
            "PASSIVE_TRACE_FILE",
            "ACTIVE_TRACE_NPY_FILE",
            "ALLEN_NWB_FILE",
            "TARGET_DESCRIPTION",
            "TARGET_NOTES",
        ):
            self.assertNotIn(old_notebook_value, setup_source)

        calls = _calls("1_setup.ipynb", "prepare_target_config")
        self.assertEqual(len(calls), 1)
        manual_toggle = _keyword(calls[0][1], "include_manual_with_file_source")
        self.assertIsInstance(manual_toggle, ast.Name)
        self.assertEqual(manual_toggle.id, "INCLUDE_MANUAL_WITH_FILE_SOURCE")

    def test_step5_uses_session_config_and_disables_unconditional_input_checks(self) -> None:
        setup_calls = _calls("5_simulate.ipynb", "finish_step5_notebook_setup")
        self.assertEqual(len(setup_calls), 1)
        check_inputs = _keyword(setup_calls[0][1], "check_external_inputs")
        self.assertIsInstance(check_inputs, ast.Constant)
        self.assertIs(check_inputs.value, False)

        mechanism_calls = _calls("5_simulate.ipynb", "ensure_modfiles")
        self.assertEqual(len(mechanism_calls), 1)
        configured_cell = _keyword(mechanism_calls[0][1], "cell_config")
        self.assertIsInstance(configured_cell, ast.Attribute)
        self.assertEqual(configured_cell.attr, "cell_config")
        self.assertIsInstance(configured_cell.value, ast.Name)
        self.assertEqual(configured_cell.value.id, "session")

        compile_value = _keyword(mechanism_calls[0][1], "compile_modfiles")
        self.assertIsInstance(compile_value, ast.Name)
        self.assertEqual(compile_value.id, "IN_COLAB")

        signature = inspect.signature(finish_step5_notebook_setup)
        self.assertIs(signature.parameters["check_external_inputs"].default, False)


if __name__ == "__main__":
    unittest.main()
