"""The contract with TypeType-Server: the algorithms the service offers and their settings."""

from typing import Literal

from . import __version__
from .contract import CONTRACT_VERSION, Model

# --- Algorithms and their parameters -------------------------------------------------


class NumberParameter(Model):
    id: str
    group: str
    type: Literal["number"] = "number"
    label: str
    help: str
    min: float
    max: float
    step: float
    default: float
    unit: str | None = None
    featured: bool = False


class BooleanParameter(Model):
    id: str
    group: str
    type: Literal["boolean"] = "boolean"
    label: str
    help: str
    default: bool
    featured: bool = False


Parameter = NumberParameter | BooleanParameter


class Traits(Model):
    personal: int
    discovery: int
    variety: int


class Guide(Model):
    for_you_if: str
    you_will_see: str
    good_to_know: str
    traits: Traits


class AlgorithmInfo(Model):
    id: str
    name: str
    description: str
    uses: list[str]
    sources: list[str]
    random: bool
    source_url: str
    guide: Guide
    parameters: list[Parameter]


class AlgorithmList(Model):
    contract_version: int = CONTRACT_VERSION
    # The version of the algorithms themselves (semantic versioning).
    version: str = __version__
    # The algorithm preselected for someone who has not chosen yet.
    default_algorithm: str
    algorithms: list[AlgorithmInfo]
