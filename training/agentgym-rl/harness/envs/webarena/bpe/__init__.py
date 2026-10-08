"""BPE (Belief / Plan / Experience) memory layer for the WebArena Claude agent.

Optional, self-contained add-on used by ``infer_claude_webarena_bpe.py``. Every
feature is gated behind a CLI flag that defaults to OFF, so importing this package
has no effect on the baseline inference loop until a flag is set.

Components
----------
- belief.py               WebWorldState: app-state graph built from the LLM
                          perception judge (no env ground truth).
- plan.py                 CommittedPlan / Subgoal (vendored, pure dataclasses).
- web_perception_judge.py No-oracle LLM judge: obs text -> progress/verified.
- web_skills_memory.py    WebSkillsMemory: retrieve + format + CRUD skills.
- skill_reflection.py     SkillReflector + SkillEvolver: post-episode evolution.
- prompt.py               Static head (General Principles) + inline per-turn blocks.

The agent issues only web actions; B/P/E are maintained in the background and
injected inline into each single-turn prompt (no meta-actions).
"""
