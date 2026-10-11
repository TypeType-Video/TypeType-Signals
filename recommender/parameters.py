"""Parameter definitions, shown to the user as-is, and how their values are read."""

from dataclasses import dataclass

from .contract_algorithms import BooleanParameter, NumberParameter, Parameter
from .text import Text, pick


@dataclass(frozen=True)
class Number:
    id: str
    group: str
    label: Text
    help: Text
    min: float
    max: float
    step: float
    default: float
    unit: str | None = None
    featured: bool = False

    def read(self, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return self.default
        return min(self.max, max(self.min, float(value)))

    def describe(self, locale: str) -> Parameter:
        return NumberParameter(
            id=self.id,
            group=self.group,
            label=pick(self.label, locale),
            help=pick(self.help, locale),
            min=self.min,
            max=self.max,
            step=self.step,
            default=self.default,
            unit=self.unit,
            featured=self.featured,
        )


@dataclass(frozen=True)
class Boolean:
    id: str
    group: str
    label: Text
    help: Text
    default: bool
    featured: bool = False

    def read(self, value: object) -> bool:
        return value if isinstance(value, bool) else self.default

    def describe(self, locale: str) -> Parameter:
        return BooleanParameter(
            id=self.id,
            group=self.group,
            label=pick(self.label, locale),
            help=pick(self.help, locale),
            default=self.default,
            featured=self.featured,
        )


Spec = Number | Boolean


class Values:
    """The user's values, clamped to each parameter's bounds, defaults for the rest."""

    def __init__(self, specs: list[Spec], given: dict[str, float | bool]):
        self._values = {spec.id: spec.read(given.get(spec.id)) for spec in specs}

    def key(self) -> tuple:
        """The values as a hashable whole (two Values with the same key decide the same)."""
        return tuple(sorted(self._values.items()))

    def number(self, id: str) -> float:
        value = self._values[id]
        assert not isinstance(value, bool), id
        return value

    def whole(self, id: str) -> int:
        return round(self.number(id))

    def flag(self, id: str) -> bool:
        value = self._values[id]
        assert isinstance(value, bool), id
        return value
