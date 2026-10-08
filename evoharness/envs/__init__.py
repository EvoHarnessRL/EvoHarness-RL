"""Registry of environment plugins, imported lazily so each env's deps stay optional."""

from __future__ import annotations

import importlib

from ..domain import Domain

DOMAINS = {
    "alfworld": "evoharness.envs.alfworld:AlfWorldDomain",
    "webshop": "evoharness.envs.webshop:WebShopDomain",
    "webarena": "evoharness.envs.webarena:WebArenaDomain",
}


def get_domain(name: str) -> Domain:
    if name not in DOMAINS:
        raise KeyError(f"unknown env {name!r}; choose from {sorted(DOMAINS)}")
    module, attr = DOMAINS[name].split(":")
    return getattr(importlib.import_module(module), attr)()
