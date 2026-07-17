from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
