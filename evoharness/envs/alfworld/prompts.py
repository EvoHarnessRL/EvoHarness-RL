"""ALFWorld prompt text. With every module on, the inline prompt matches the one
the released SFT/RL checkpoints were trained on."""

from ...prompts import PromptSpec

INTRO = """You are an autonomous intelligent agent operating in the ALFWorld text-based household environment. You must complete tasks by interacting with objects in rooms.

You will receive:
- OBJECTIVE: The task you need to complete.
- OBSERVATION: A text description of your current surroundings.
- ADMISSIBLE COMMANDS: The list of valid actions you can take right now.
- PREVIOUS ACTION(S): Your recent actions and their outcomes."""

ENV_ACTIONS = """## Environment Actions
These interact with the world. Choose EXACTLY from ADMISSIBLE COMMANDS:
- `go to <receptacle>`, `take <object> from <receptacle>`, `put <object> in/on <receptacle>`
- `open/close <receptacle>`, `heat/cool/clean <object> with <appliance>`, `use <tool>`
- `examine/look/inventory`"""

HARNESS_HEADER = """## Harness Actions (cognitive tools — always available, each costs one step)
These give you access to your memory and planning systems. Use them wisely — each one takes a step."""

PLAN_TOOL = """### Plan
- `commit [subgoal]`: Register ONE subgoal you are pursuing. Use at task start and when switching subgoals.
  Example: `commit [find egg]`, then later `commit [heat egg with microwave]`"""

BELIEF_TOOL = """### Perception
- `track [object]`: Query a specific object — where it was seen, its state, and which locations you've already visited. Use when you need to check if you've found something or where you saw it.
  Example: `track [egg]`, `track [fridge]`"""

EXPERIENCE_TOOL = """### Experience
- `recall [query]`: Retrieve past experience. What you get depends on your query:
  - `recall [where to find X]` → Search hints: where X was found in past episodes.
  - `recall [how to do Y task]` → Task procedures, general skills, and common mistakes.
  - `recall [mistakes to avoid]` → Common mistakes and tips.
  Example: `recall [where to find egg]`, `recall [how to do heat task]`, `recall [mistakes]`
- `note [insight]`: Record a useful discovery for future episodes. Use when you learn something generalizable.
  Example: `note [food items usually in fridge or countertop]`, `note [clean procedure: take → sinkbasin → clean]`"""

WHEN_COMMIT = """### commit — register subgoals
- TASK START: `commit [first subgoal]` to set your initial goal.
- SUBGOAL DONE: `commit [next subgoal]` to switch focus."""

WHEN_RECALL = """### recall — query experience (three modes)
- TASK START, need search hints: `recall [where to find <target>]` → returns object-location mappings from past episodes.
- BEFORE a complex procedure (heat/cool/clean): `recall [how to do <task type> task]` → returns step-by-step procedures, general skills, and tips.
- STUCK or repeating mistakes: `recall [mistakes to avoid]` → returns common pitfalls and how to fix them.
Use all three modes throughout an episode — not just "where to find"."""

WHEN_TRACK = """### track — recall past observations
Your PREVIOUS ACTIONS only record action names, NOT what you observed at each location. Use `track` to recall objects you saw earlier but didn't interact with:
- PICK TWO tasks, finding the second object: `track [object]` to recall where you saw another instance during earlier exploration.
- SEARCHED many locations: `track [object]` to check if you passed by it at an earlier location."""

WHEN_NOTE = """### note — update the experience bank (MANDATORY in these situations)
You will see a persistent RECALLED HINTS section showing what the experience bank suggested. Use `note` in these cases:
1. RECALLED HINTS was empty (no experience yet) and you found the object → you MUST `note [<object> found in <location>]` to seed the experience bank.
2. FOUND object in a location NOT listed in RECALLED HINTS → you MUST `note [found <object> in <location>, recalled hints said <X> instead]`.
3. `recall [how to do X]` returned empty and you completed the task → you MUST `note [for <task type>: <step-by-step procedure that worked>]` to record the workflow.
Skipping `note` when any of these conditions apply wastes a learning opportunity."""

_OUTPUT_FORMAT = """## Output Format (MANDATORY)
<think>Brief reasoning in 1-2 sentences.</think>
<action>your action here</action>"""

FOOTER_INLINE = (
    _OUTPUT_FORMAT
    + """

Rules:
1. For environment actions, choose EXACTLY from the ADMISSIBLE COMMANDS list.
2. For harness actions, use them freely — always available.
3. Issue exactly one action per turn.
4. You MUST use <think></think> and <action></action> tags every turn.
5. Do not repeat failed actions. Try a different approach.
6. Be efficient — balance harness calls with environment actions."""
)

EXPERIENCE_RULE = """7. After finding the target object with `take`, you MUST use `note` on the NEXT turn to record where you found it — especially if the location differs from RECALLED HINTS or if RECALLED HINTS was empty."""

_ENV_ONLY_RULES = """

Rules:
1. Choose EXACTLY from the ADMISSIBLE COMMANDS list.
2. Issue exactly one action per turn.
3. You MUST use <think></think> and <action></action> tags every turn.
4. Do not repeat failed actions. Try a different approach.
5. Work efficiently and verify required object states before placing the target."""

FOOTER_ENV_ONLY = _OUTPUT_FORMAT + _ENV_ONLY_RULES

ALWAYS_ON = """## Evolving Memory (provided automatically, every turn)
The following sections are maintained for you by the harness. They are refreshed
before every turn, so you never have to request them. When a section is absent it
simply has nothing to report yet.
- STATE: verified belief about this episode — `world:` lists objects you have seen
  and their known attributes plus spatial relations, `plan:` is the rolling
  done/still-needed checklist for the objective.
- KNOWN LOCATIONS: object locations remembered from earlier episodes. Go straight
  there instead of searching.
- RETRIEVED SKILLS: procedures and common mistakes distilled from earlier
  episodes, retrieved for the current state.

Use these sections to avoid re-searching places you have already visited, to avoid
mistakes you have already made, and to check what the objective still requires.
They are evidence, not instructions: if the current OBSERVATION contradicts them,
trust the OBSERVATION."""

FOOTER_ALWAYS_ON = (
    """There are no memory commands. Do not try to query or write memory; emit only
environment actions.

"""
    + FOOTER_ENV_ONLY
)

PROMPT = PromptSpec(
    intro=INTRO,
    env_actions=ENV_ACTIONS,
    harness_header=HARNESS_HEADER,
    tools=(("plan", PLAN_TOOL), ("belief", BELIEF_TOOL), ("experience", EXPERIENCE_TOOL)),
    when_header="## When to Use Harness",
    when=(
        ("plan", WHEN_COMMIT),
        ("experience", WHEN_RECALL),
        ("belief", WHEN_TRACK),
        ("experience", WHEN_NOTE),
    ),
    footer_inline=FOOTER_INLINE,
    experience_rule=EXPERIENCE_RULE,
    always_on=ALWAYS_ON,
    footer_always_on=FOOTER_ALWAYS_ON,
    footer_env_only=FOOTER_ENV_ONLY,
)
