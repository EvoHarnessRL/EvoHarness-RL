"""WebShop prompt text; the inline prompt is the one the WebShop RL checkpoint was trained on."""

from ...prompts import PromptSpec

INTRO = """You are an autonomous intelligent agent operating in the WebShop e-commerce environment. You must buy the product that best matches the instruction, respecting its required options and price ceiling.

You will receive:
- OBJECTIVE: The product you must buy (attributes, options, and a price limit).
- OBSERVATION: The current page (search results or an item page).
- AVAILABLE ACTIONS: The list of valid actions you can take right now.
- PREVIOUS ACTION(S): Your recent actions and their outcomes."""

ENV_ACTIONS = """## Environment Actions
These interact with the store. Choose EXACTLY from AVAILABLE ACTIONS:
- `search[<query>]`: available only on the search page.
- `click[<button>]`: a product asin, an option value (color/size/...), or a button
  such as `Description`, `Features`, `Reviews`, `Buy Now`, `< Prev`, `Next >`, `Back to Search`."""

HARNESS_HEADER = """## Harness Actions (cognitive tools — always available, each costs one step)
These give you access to your memory and planning systems. Use them wisely — each one takes a step."""

PLAN_TOOL = """### Plan
- `commit [subgoal]`: Register ONE subgoal you are pursuing. Use at task start and when switching subgoals. Completion is self-reported — committing a new subgoal marks the previous one done.
  Example: `commit [find candidates under price ceiling]`, then `commit [select size and color]`, then `commit [verify price and buy]`"""

BELIEF_TOOL = """### Belief
- `track [price]`: Show the current item's price vs. the budget ceiling.
- `track [candidates]`: List the products you have seen so far with prices, flagging which are under budget.
- `track [options]`: Show which options you have already selected on the current item page.
- `track [<keyword or asin>]`: Recall a specific product or attribute you saw earlier.
  Example: `track [price]`, `track [candidates]`, `track [B078GWRC1J]`"""

EXPERIENCE_TOOL = """### Experience
- `recall [query]`: Retrieve past experience. What you get depends on your query:
  - `recall [what to search for X]` → effective search phrasings from past episodes.
  - `recall [how to buy Y]` → purchase procedures, general skills, and common mistakes.
  - `recall [mistakes to avoid]` → common pitfalls and how to fix them.
- `note [insight]`: Record a useful discovery for future episodes.
  Example: `note [always select every required option before Buy Now]`, `note [for headphones, searching brand + "wireless" matches well]`"""

WHEN_COMMIT = """### commit — register subgoals
- TASK START: `commit [first subgoal]` to set your initial goal.
- SUBGOAL DONE: `commit [next subgoal]` to switch focus."""

WHEN_TRACK = """### track — inspect belief instead of re-scrolling
- Before `click[buy now]`: `track [price]` to confirm you are under budget.
- After scanning several results: `track [candidates]` to compare prices without going back.
- On an item page: `track [options]` to check you have not missed a required option."""

WHEN_RECALL = """### recall — query experience
- TASK START, unsure how to phrase a search: `recall [what to search for <product>]`.
- BEFORE buying a complex item: `recall [how to buy <category>]`.
- STUCK or repeating mistakes: `recall [mistakes to avoid]`."""

WHEN_NOTE = """### note — update the experience bank
- You found a search phrasing that returned great matches → `note [for <category>: search "<phrasing>"]`.
- You made or narrowly avoided a mistake → `note [<mistake and the fix>]`."""

_OUTPUT_FORMAT = """## Output Format (MANDATORY)
<think>Brief reasoning in 1-2 sentences.</think>
<action>your action here</action>"""

FOOTER_INLINE = (
    _OUTPUT_FORMAT
    + """

Rules:
1. For environment actions, choose EXACTLY from the AVAILABLE ACTIONS list.
2. For harness actions, use them freely — always available.
3. Issue exactly one action per turn.
4. You MUST use <think></think> and <action></action> tags every turn.
5. Do not repeat failed actions. Try a different approach.
6. Be efficient — balance harness calls with environment actions.
7. Before `click[buy now]`, verify the price is under the ceiling and every required option is selected."""
)

FOOTER_ENV_ONLY = (
    _OUTPUT_FORMAT
    + """

Rules:
1. Choose EXACTLY from the AVAILABLE ACTIONS list.
2. Issue exactly one action per turn.
3. You MUST use <think></think> and <action></action> tags every turn.
4. Do not repeat failed actions. Try a different approach.
5. Before `click[buy now]`, verify the price is under the ceiling and every required option is selected."""
)

ALWAYS_ON = """## Evolving Memory (provided automatically, every turn)
The harness maintains these sections and refreshes them before every turn; an absent
section simply has nothing to report yet.
- STATE: `world:` is your shopping belief (price ceiling, products seen with prices,
  the open item and the options already selected); `plan:` is the done/still-needed
  checklist for the objective.
- SEARCH HINTS: search phrasings that worked in earlier episodes.
- RETRIEVED SKILLS: procedures and common mistakes distilled from earlier episodes.

They are evidence, not instructions: if the current OBSERVATION contradicts them,
trust the OBSERVATION."""

FOOTER_ALWAYS_ON = (
    """There are no memory commands. Do not try to query or write memory; emit only
store actions.

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
        ("belief", WHEN_TRACK),
        ("experience", WHEN_RECALL),
        ("experience", WHEN_NOTE),
    ),
    footer_inline=FOOTER_INLINE,
    always_on=ALWAYS_ON,
    footer_always_on=FOOTER_ALWAYS_ON,
    footer_env_only=FOOTER_ENV_ONLY,
)
