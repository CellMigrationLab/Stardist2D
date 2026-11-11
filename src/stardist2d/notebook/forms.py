"""Forms that replace the Colab ``#@param`` fields, built from core ipywidgets only.

Colab forms do not exist in JupyterLab. Each notebook step shows one :class:`Form`: the fields of the
original cell, with the same names and defaults, and a button that runs the step. Only widgets and
options present in both ipywidgets 7.7 (Colab) and 8 (LabConstrictor) are used.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import ipywidgets as widgets
from IPython.display import display

_STYLE = {"description_width": "260px"}
_LAYOUT = {"width": "760px"}
EXPECTED_PROBLEMS = (ValueError, FileNotFoundError, FileExistsError, RuntimeError)
"""Problems the person can fix: shown as one red line, without a traceback."""


@dataclass(frozen=True)
class Field:
    """One form field: the 1.20.8 parameter *name*, its default and how it is edited.

    ``kind`` is ``text``, ``int``, ``float``, ``bool``, ``choice`` (with ``options``) or ``slider``
    (with ``options = (min, max, step)``).
    """

    name: str
    default: Any
    kind: str = "text"
    options: Sequence[Any] = ()
    help: str = ""

    def widget(self) -> widgets.Widget:
        """Return the widget editing this field."""
        common: dict[str, Any] = {"description": self.name, "style": _STYLE, "layout": _LAYOUT}
        if self.help:
            common["tooltip"] = self.help
        if self.kind == "bool":
            return widgets.Checkbox(value=bool(self.default), indent=False, **common)
        if self.kind == "int":
            return widgets.IntText(value=int(self.default), **common)
        if self.kind == "float":
            return widgets.FloatText(value=float(self.default), **common)
        if self.kind == "choice":
            return widgets.Dropdown(options=list(self.options), value=self.default, **common)
        if self.kind == "slider":
            low, high, step = self.options
            return widgets.IntSlider(value=int(self.default), min=low, max=high, step=step, **common)
        return widgets.Text(value=str(self.default), **common)


@dataclass
class Form:
    """A titled set of fields, a run button and an output area; ``run`` calls the step with the values."""

    title: str
    fields: Sequence[Field]
    action: Callable[[dict[str, Any]], None]
    button_label: str = "Run"
    controls: dict[str, widgets.Widget] = field(init=False)
    button: widgets.Button = field(init=False)
    output: widgets.Output = field(init=False)
    succeeded: bool | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.controls = {spec.name: spec.widget() for spec in self.fields}
        self.button = widgets.Button(description=self.button_label, button_style="primary", icon="play")
        self.output = widgets.Output()
        self.button.on_click(lambda _: self.run())

    def values(self) -> dict[str, Any]:
        """Return the current value of every field, by parameter name."""
        return {name: control.value for name, control in self.controls.items()}

    def set(self, **values: object) -> Form:
        """Set fields by name (used by tests and scripted runs) and return the form."""
        for name, value in values.items():
            self.controls[name].value = value
        return self

    def run(self) -> bool:
        """Run the step with the current values; show expected problems as one line. Return success."""
        self.output.clear_output()
        with self.output:
            try:
                self.action(self.values())
            except EXPECTED_PROBLEMS as problem:
                display(widgets.HTML(f"<b style='color:#b00020'>⚠ {problem}</b>"))
                self.succeeded = False
                return False
        self.succeeded = True
        return True

    def box(self) -> widgets.VBox:
        """Return the form as one widget."""
        title = widgets.HTML(f"<h4 style='margin:4px 0'>{self.title}</h4>")
        return widgets.VBox([title, *self.controls.values(), self.button, self.output])

    def show(self) -> Form:
        """Display the form and return it."""
        display(self.box())
        return self


def message(text: str, warning: bool = False) -> None:
    """Show a short message (orange for a warning, as 1.20.8's ``bcolors.WARNING``)."""
    colour = "#b36b00" if warning else "#1b5e20"
    display(widgets.HTML(f"<span style='color:{colour}'>{text}</span>"))
