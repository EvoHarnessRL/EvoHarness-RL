"""ALFWorld (TextWorld) environment adapter. Requires ``pip install alfworld`` and ``$ALFWORLD_DATA``."""

from __future__ import annotations

import functools
import re
import threading
from pathlib import Path

import yaml

from ...domain import Step, Task

DEFAULT_CONFIG = str(Path(__file__).with_name("config_tw.yaml"))
SPLITS = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
    "eval_in_distribution": "eval_in_distribution",
    "eval_out_of_distribution": "eval_out_of_distribution",
}
FAMILIES = (
    ("pick_two_obj_and_place", "pick_two"),
    ("pick_clean_then_place_in_recep", "clean"),
    ("pick_heat_then_place_in_recep", "heat"),
    ("pick_cool_then_place_in_recep", "cool"),
    ("look_at_obj_in_light", "look"),
    ("pick_and_place_simple", "pick"),
)

# TextWorld's PDDL parser is not thread-safe; every env call goes through this lock.
_LOCK = threading.Lock()


def _load_config(config_path: str | None) -> dict:
    with open(config_path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


@functools.cache
def list_game_files(split: str, config_path: str | None = None) -> tuple[str, ...]:
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    if split not in SPLITS:
        raise ValueError(f"ALFWorld split must be one of {sorted(SPLITS)}")
    with _LOCK:
        env = AlfredTWEnv(_load_config(config_path), train_eval=SPLITS[split])
    return tuple(sorted(str(Path(p).resolve()) for p in env.game_files))


def task_id(game_file: str) -> str:
    path = Path(game_file)
    return f"{path.parent.parent.name}/{path.parent.name}"


def family(game_file: str) -> str:
    folder = Path(game_file).parent.parent.name
    return next((short for prefix, short in FAMILIES if folder.startswith(prefix)), "unknown")


class AlfWorldEnv:
    def __init__(self, config_path: str | None = None):
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

        with _LOCK:
            # Any split works as a loader; reset() points it at the requested game.
            self._loader = AlfredTWEnv(_load_config(config_path), train_eval="eval_in_distribution")
        self._env = None
        self._last = Step(observation="")

    def reset(self, task: Task) -> Step:
        with _LOCK:
            self.close()
            self._loader.game_files = [task.data]
            self._loader.num_games = 1
            self._env = self._loader.init_env(batch_size=1)
            obs, info = self._env.reset()
        observation = _strip_banner(obs[0])
        match = re.search(r"Your task is to:\s*(.+?)\.?\s*$", observation)
        self._last = Step(
            observation=observation,
            actions=_admissible(info),
            objective=match.group(1).strip() if match else "complete the task",
            info={"game_file": task.data, "task_type": family(task.data)},
        )
        return self._last

    def step(self, action: str) -> Step:
        command = next((c for c in self._last.actions if c.lower() == action.strip().lower()), None)
        if command is None:
            self._last = Step(
                observation=self._last.observation,
                actions=self._last.actions,
                objective=self._last.objective,
                valid=False,
            )
            return self._last
        with _LOCK:
            obs, _, dones, info = self._env.step([command])
        won = bool(_first(info.get("won")))
        self._last = Step(
            observation=obs[0],
            actions=_admissible(info),
            objective=self._last.objective,
            done=bool(dones[0]) or won,
            won=won,
            score=float(won),
        )
        return self._last

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001 - textworld close is best-effort
                pass
            self._env = None


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) and value else value


def _admissible(info: dict) -> list[str]:
    return [c for c in _first(info.get("admissible_commands")) or [] if c != "help"]


def _strip_banner(observation: str) -> str:
    return re.sub(r"^-= Welcome to TextWorld, ALFRED! =-\s*", "", observation)
