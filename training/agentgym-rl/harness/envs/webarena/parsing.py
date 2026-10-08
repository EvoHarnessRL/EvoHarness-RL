"""WebArena-specific prompt constant and observation/action parsing helpers.

Extracted verbatim from ``infer/infer_claude_webarena_bpe.py``. These are the
web-navigation instantiation details: the system prompt / action grammar, the
accessibility-tree observation parsing, the single-turn frame helpers, and the
per-turn skill-retrieval query builder. Behavior unchanged.
"""
import re


def parse_action_verb(action: str) -> str:
    """First token of a parsed action (content inside the ```...``` fence)."""
    if not isinstance(action, str):
        return ""
    return action.strip().split("[", 1)[0].strip().split(" ", 1)[0].strip().lower()


SYSTEM_PROMPT = (
    "You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. "
    "These tasks will be accomplished through the use of specific actions you can issue.\n\n"
    "Here's the information you'll have:\n"
    "The user's objective: This is the task you're trying to complete.\n"
    "The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.\n"
    "The current web page's URL: This is the page you're currently navigating.\n"
    "The open tabs: These are the tabs you have open.\n"
    "The previous action: This is the action you just performed. It may be helpful to track your progress.\n\n"
    "The actions you can perform fall into several categories:\n\n"
    "Page Operation Actions:\n"
    "```click [id]```: This action clicks on an element with a specific id on the webpage.\n"
    "```type [id] [content] [0|1]```: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless the last parameter is set to 0.\n"
    "```hover [id]```: Hover over an element with id.\n"
    "`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n"
    "```scroll [down|up]```: Scroll the page up or down to reveal content below or above the current view.\n\n"
    "Tab Management Actions:\n"
    "```new_tab```: Open a new, empty browser tab.\n"
    "```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index.\n"
    "```close_tab```: Close the currently active tab.\n\n"
    "URL Navigation Actions:\n"
    "```goto [url]```: Navigate to a specific URL.\n"
    "```go_back```: Navigate to the previously viewed page.\n"
    "```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).\n\n"
    "Completion Action:\n"
    "```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as \"N/A\" in the bracket.\n\n"
    "Homepage:\n"
    "If you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.\n\n"
    "To be successful, it is very important to follow the following rules:\n"
    "1. You should only issue an action that is valid given the current observation\n"
    "2. You should only issue one action at a time.\n"
    "3. You should follow the examples to reason step by step and then issue the next action.\n"
    "4.For ALL actions that take parameters, you MUST enclose each parameter in square brackets [].\n"
    "5. Generate the action in the correct format. Start with a \"Let's think step-by-step...In summary, the next action I will perform is\" phrase, followed by action inside triple backticks (```). For example, \"Let's think step-by-step. This page has a search box whose ID is [164]. According to the nominatim rule of openstreetmap, I can search for the restaurants near a location by \"restaurants near\". I can submit my typing by pressing the Enter afterwards. In summary, the next action I will perform is ```type [164] [restaurants near CMU] [1]```\".\n"
    "6. Issue stop action when you think you have achieved the objective. Don't generate anything after stop."
)


def split_reasoning_action(text):
    """Split model output into (reasoning, action). Action = content of first ```...```."""
    m = re.search(r"```(.*?)```", text, re.DOTALL)
    action = m.group(1).strip() if m else ""
    if m:
        reasoning = text[:m.start()].strip()
    else:
        reasoning = text.strip()
    # drop the trailing boilerplate lead-in if present
    reasoning = re.sub(r"In summary,?\s*the next action I will perform is\s*$", "",
                       reasoning, flags=re.IGNORECASE).strip()
    return reasoning, action


def extract_objective(obs_text):
    m = re.search(r"OBJECTIVE:\s*(.*?)(?:\nPREVIOUS ACTION:|\Z)", obs_text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_url(obs_text):
    m = re.search(r"URL:\s*(\S+)", obs_text)
    return m.group(1) if m else ""


# --------------------------------------------------------------------------- #
# Single-turn frame helpers (shared design with infer_claude_webarena.py)      #
# --------------------------------------------------------------------------- #
def truncate_frame(frame, max_obs_chars):
    """Cap the (big) accessibility-tree part of the frame while keeping the
    trailing URL / OBJECTIVE / PREVIOUS ACTION lines intact."""
    if not max_obs_chars or max_obs_chars <= 0 or len(frame) <= max_obs_chars:
        return frame
    tail_i = frame.find("\nURL:")
    if tail_i < 0:
        return frame[:max_obs_chars] + "\n... [observation truncated] ..."
    if tail_i <= max_obs_chars:
        return frame
    return (frame[:max_obs_chars] + "\n... [observation truncated] ...\n" + frame[tail_i:])


def strip_prev_action_line(frame):
    """Remove the env frame's single-action 'PREVIOUS ACTION:' tail; we render the
    full compact action history ourselves instead."""
    i = frame.find("\nPREVIOUS ACTION:")
    return frame[:i] if i >= 0 else frame


def render_history(action_hist, keep=10):
    """Compact numbered trace of the recent actions (no bulky observations)."""
    if not action_hist:
        return "None"
    recent = action_hist[-keep:]
    start = len(action_hist) - len(recent) + 1
    return "\n".join(f"{start+i}. {a}" for i, a in enumerate(recent))


def build_skill_query(intent, plan, last_action, belief=None):
    """Per-turn ADAPTIVE retrieval query: re-composed every step from the live
    state so the skills retrieved track the agent's current situation, not just
    the static objective. Combines: objective + what the plan still needs + the
    last action + the current page/site (from belief)."""
    parts = [intent]
    if plan is not None:
        sg = plan.active_subgoal() or (plan.items[-1] if plan.items else None)
        if sg and sg.missing:
            parts.extend(sg.missing[:2])
    if last_action:
        parts.append(last_action)
    if belief is not None:
        pages = [n.name for n in belief.nodes.values() if n.role == "page"]
        if pages:
            parts.append(pages[-1])  # the page the agent is currently on
    return " ".join(p for p in parts if p)


def belief_notes(belief):
    """Auto-collected data records from the belief, used as reflector notes
    (replaces the agent-issued ``note[...]`` in the pure-background design)."""
    if belief is None:
        return []
    notes = []
    for name, node in belief.nodes.items():
        if node.role == "data_record":
            val = node.attrs.get("value")
            nt = node.attrs.get("note", "")
            if val:
                notes.append(f"{name}: {val}" + (f" ({nt})" if nt else ""))
    return notes[:20]
