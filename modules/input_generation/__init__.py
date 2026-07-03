"""Input generation package: config normalization, input modes, and sampling."""

from .inputs import GroupInputs, check_inputs, generate_inputs
from .modes_core import get_default_mode_registry
from .modes_user import get_user_mode_registry
from .sampling import sample_group_rates, sample_group_rates_from_path

__all__ = [
    "GroupInputs",
    "check_inputs",
    "generate_inputs",
    "get_default_mode_registry",
    "get_user_mode_registry",
    "sample_group_rates",
    "sample_group_rates_from_path",
]
