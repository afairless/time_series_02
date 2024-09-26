#! /usr/bin/env python3

from typing import Any


class VariableTracker:

    def __init__(self) -> None:
        self.data = {}

    def __setitem__(self, key, value) -> None:
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def __getitem__(self, key) -> Any:
        return self.data[key][-1] if key in self.data else None

    def get_all_vars_all_values(self) -> dict[Any, Any]:
        return self.data

    def get_all_var_values(self, key) -> list[Any]:
        return self.data.get(key, [])


def track_global_variables(code: str) -> dict[Any, Any]:

    tracker = VariableTracker()

    class CustomLocals(dict):
        def __setitem__(self, key, value):
            tracker[key] = value
            super().__setitem__(key, value)

    custom_locals = CustomLocals()

    exec(code, {'__builtins__': __builtins__}, custom_locals)

    return tracker.get_all_vars_all_values()


