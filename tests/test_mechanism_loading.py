from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from modules.setup import mechanisms
from modules.setup.mechanisms import load_compiled_mechanism_library


class MechanismLoadingTests(unittest.TestCase):
    def test_same_path_and_hash_is_skipped_but_changed_library_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dll = Path(tmp) / "libnrnmech.so"
            dll.write_bytes(b"synthetic-dll-v1")
            fake_h = mock.Mock()
            fake_h.nrn_load_dll.return_value = 1
            fake_neuron = types.SimpleNamespace(h=fake_h)
            with mock.patch.dict("sys.modules", {"neuron": fake_neuron}):
                first = load_compiled_mechanism_library(dll)
                second = load_compiled_mechanism_library(dll)
            self.assertTrue(first["loaded"])
            self.assertFalse(first["dll_preloaded"])
            self.assertTrue(second["dll_preloaded"])
            fake_h.nrn_load_dll.assert_called_once_with(str(dll.resolve()))

            dll.write_bytes(b"synthetic-dll-v2")
            with self.assertRaisesRegex(RuntimeError, "changed after it was loaded"):
                load_compiled_mechanism_library(dll)

    def test_neuron_load_failure_is_not_silently_treated_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dll = Path(tmp) / "conflicting.so"
            dll.write_bytes(b"synthetic-conflict")
            fake_h = mock.Mock()
            fake_h.nrn_load_dll.side_effect = RuntimeError("collision")
            fake_neuron = types.SimpleNamespace(h=fake_h)
            with mock.patch.dict("sys.modules", {"neuron": fake_neuron}):
                with self.assertRaisesRegex(RuntimeError, "Restart"):
                    load_compiled_mechanism_library(dll)

    def test_recompile_refuses_to_remove_a_loaded_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            mod_dir = tune / "modfiles"
            dll = mod_dir / "x86_64" / ".libs" / "libnrnmech.so"
            dll.parent.mkdir(parents=True)
            dll.write_bytes(b"synthetic-loaded-dll")
            (mod_dir / "synthetic.mod").write_text("TITLE synthetic\n")
            digest = mechanisms.mechanism_dll_sha256(dll)

            with mock.patch.dict(
                mechanisms._LOADED_MECHANISM_DLLS,
                {dll.resolve(): digest},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "Restart"):
                    mechanisms.compile_modfiles(
                        tune,
                        recompile=True,
                        load_dll=False,
                    )

            self.assertTrue(dll.is_file())

    def test_recompile_rebuilds_an_unloaded_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tune = Path(tmp)
            mod_dir = tune / "modfiles"
            dll = mod_dir / "x86_64" / ".libs" / "libnrnmech.so"
            dll.parent.mkdir(parents=True)
            dll.write_bytes(b"stale-dll")
            (mod_dir / "synthetic.mod").write_text("TITLE synthetic\n")

            def build_library(*_args, **_kwargs):
                dll.parent.mkdir(parents=True)
                dll.write_bytes(b"rebuilt-dll")

            with (
                mock.patch(
                    "modules.setup.mechanisms.find_nrnivmodl",
                    return_value="/synthetic/nrnivmodl",
                ),
                mock.patch(
                    "modules.setup.mechanisms.subprocess.check_call",
                    side_effect=build_library,
                ) as check_call,
            ):
                summary = mechanisms.compile_modfiles(
                    tune,
                    recompile=True,
                    load_dll=False,
                )

            check_call.assert_called_once_with(
                ["/synthetic/nrnivmodl"],
                cwd=str(mod_dir.resolve()),
            )
            self.assertTrue(summary["compiled_now"])
            self.assertEqual(dll.read_bytes(), b"rebuilt-dll")


if __name__ == "__main__":
    unittest.main()
