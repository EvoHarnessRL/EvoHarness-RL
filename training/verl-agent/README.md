# verl-agent (training path A)

The verl-agent training framework as used by SkillRL, which our ALFWorld and
WebShop GRPO runs were trained with. It is built on
[verl-agent](https://github.com/langfengQ/verl-agent) and
[verl](https://github.com/volcengine/verl). Only the modules training needs are
included: `verl/`, `gigpo/`, and the rollout and reward parts of `agent_system/`.

Install into the training environment with
`pip install -e training/verl-agent --no-deps` (dependencies:
`requirements-train.txt` at the repository root). The entry point is
`python -m evoharness.rl.train`; see the main README.
