from __future__ import annotations

import ast
import inspect
import json
import unittest
from pathlib import Path
from typing import Iterable

from modules.notebooks.bootstrap import finish_step5_notebook_setup


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _assigned_dict(notebook_name: str, variable_name: str) -> tuple[str, ast.Dict]:
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
            if not isinstance(node.value, ast.Dict):
                raise AssertionError(f"{variable_name} must be assigned a dict literal")
            return source, node.value
    raise AssertionError(f"No assignment to {variable_name!r} found in {notebook_name}")


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
        self.assertIn("HOC_CONSTRUCTOR_ARGS = None", setup_source)
        self.assertIn("HOC_SECTION_MAP = None", setup_source)
        self.assertIn("if HOC_TEMPLATE_FILE not in (None, \"\"):", setup_source)
        self.assertIn("if HOC_CONSTRUCTOR_ARGS is not None:", setup_source)
        self.assertNotIn(
            'loader_paths = {"hoc_template": HOC_TEMPLATE_FILE, "modfiles": HOC_MODFILES_DIR}',
            setup_source,
        )

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
