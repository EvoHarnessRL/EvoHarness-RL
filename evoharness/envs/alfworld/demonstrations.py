"""The seven worked ALFWorld turns used as SFT few-shots.

Stored as :class:`~evoharness.domain.Demonstration` inputs rather than rendered
text, so they are always formatted exactly like the turns they precede. The
originals were written against a slightly different renderer (unquoted command
lists, JSON-encoded history); rendering them through ``render_turn`` keeps the
reasoning verbatim while making the formatting match what the student will see.

Switch them off with ``sft.few_shots=false`` to train on bare turns.
"""

from ...domain import Demonstration

DEMONSTRATIONS = (
    Demonstration(
        objective="clean some mug and put it in shelf.",
        observation="You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a coffeemachine 1, a countertop 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 1, a sinkbasin 1, a stoveburner 1.",
        actions=(
            "look",
            "inventory",
            "go to cabinet 1",
            "go to coffeemachine 1",
            "go to countertop 1",
            "go to drawer 1",
            "go to fridge 1",
            "go to garbagecan 1",
            "go to microwave 1",
            "go to shelf 1",
            "go to sinkbasin 1",
            "go to stoveburner 1",
        ),
        response="""<think>New task: clean mug and place in shelf. This is a clean task so I should recall the procedure before starting.</think>
<action>commit [find mug]</action>""",
    ),
    Demonstration(
        objective="clean some mug and put it in shelf.",
        observation="You are in the middle of a room.",
        actions=(
            "look",
            "inventory",
            "go to cabinet 1",
            "go to coffeemachine 1",
            "go to countertop 1",
            "go to drawer 1",
            "go to fridge 1",
            "go to garbagecan 1",
            "go to microwave 1",
            "go to shelf 1",
            "go to sinkbasin 1",
            "go to stoveburner 1",
        ),
        history=(
            "commit [find mug] → (plan updated)",
        ),
        views="""PLAN:
- [ ] find mug (open)""",
        response="""<think>Let me recall the procedure for clean tasks so I know the full workflow.</think>
<action>recall [how to do clean task]</action>""",
    ),
    Demonstration(
        objective="find two pillow and put them in sofa.",
        observation="You move the pillow 1 to the sofa 1.",
        actions=(
            "examine sofa 1",
            "go to armchair 1",
            "go to bed 1",
            "go to desk 1",
            "go to drawer 1",
            "go to drawer 2",
            "go to dresser 1",
            "go to garbagecan 1",
            "go to sidetable 1",
        ),
        history=(
            "...",
            "take pillow 1 from bed 1",
            "go to sofa 1",
            "move pillow 1 to sofa 1",
        ),
        views="""PLAN:
- [x] find and place first pillow (complete)
- [ ] find second pillow (open)""",
        response="""<think>First pillow done. I need the second one. Let me track to recall where I saw another pillow during my earlier exploration.</think>
<action>track [pillow]</action>""",
    ),
    Demonstration(
        objective="find two pillow and put them in sofa.",
        observation="You move the pillow 1 to the sofa 1.",
        actions=(
            "examine sofa 1",
            "go to armchair 1",
            "go to bed 1",
            "go to desk 1",
            "go to drawer 1",
            "go to drawer 2",
            "go to dresser 1",
            "go to garbagecan 1",
            "go to sidetable 1",
        ),
        history=(
            "...",
            "move pillow 1 to sofa 1",
            "track [pillow] → (state checked)",
        ),
        views="""TRACKED:
- pillow 2 (other): unknown state
- pillow 1 (target): isPickedUp=True
relations:
  pillow 2 on armchair 1
  pillow 1 on sofa 1

PLAN:
- [x] find and place first pillow (complete)
- [ ] find second pillow (open)""",
        response="""<think>Track shows pillow 2 is on armchair 1. I saw it earlier but didn't pick it up. Going there now.</think>
<action>go to armchair 1</action>""",
    ),
    Demonstration(
        objective="put a hot apple in fridge.",
        observation="You open the drawer 1. The drawer 1 is open. In it, you see nothing.",
        actions=(
            "close drawer 1",
            "go to cabinet 1",
            "go to countertop 1",
            "go to diningtable 1",
            "go to fridge 1",
            "go to garbagecan 1",
            "go to microwave 1",
            "go to sinkbasin 1",
            "go to stoveburner 1",
        ),
        history=(
            "...",
            "recall [where to find apple] → (skills retrieved)",
            "go to countertop 1",
            "go to diningtable 1",
            "go to drawer 1",
            "open drawer 1",
        ),
        views="""PLAN:
- [ ] find apple (open)""",
        response="""<think>Searched countertop, diningtable, and drawer — no apple. The recall hints were wrong. Let me check what mistakes I should avoid before wasting more steps.</think>
<action>recall [mistakes to avoid]</action>""",
    ),
    Demonstration(
        objective="put a hot apple in fridge.",
        observation="You arrive at garbagecan 1. On the garbagecan 1, you see a apple 1.",
        actions=(
            "examine garbagecan 1",
            "take apple 1 from garbagecan 1",
            "go to cabinet 1",
            "go to countertop 1",
            "go to diningtable 1",
            "go to drawer 1",
            "go to fridge 1",
            "go to microwave 1",
            "go to sinkbasin 1",
            "go to stoveburner 1",
        ),
        history=(
            "...",
            "recall [where to find apple] → (skills retrieved)",
            "go to countertop 1",
            "go to diningtable 1",
            "go to garbagecan 1",
        ),
        views="""PLAN:
- [ ] find apple (open)

RECALLED HINTS:
- apple: countertop, diningtable""",
        response="""<think>Found apple in garbagecan, but RECALLED HINTS said countertop and diningtable. I must note this to correct the experience bank.</think>
<action>note [apple found in garbagecan, recalled hints said countertop and diningtable instead]</action>""",
    ),
    Demonstration(
        objective="clean some plate and put it in diningtable.",
        observation="You move the plate 1 to the diningtable 1.",
        actions=(
            "examine diningtable 1",
            "go to cabinet 1",
            "go to countertop 1",
            "go to fridge 1",
            "go to sinkbasin 1",
        ),
        history=(
            "...",
            "clean plate 1 with sinkbasin 1",
            "go to diningtable 1",
            "move plate 1 to diningtable 1",
        ),
        views="""PLAN:
- [x] find plate (complete)
- [x] clean plate with sinkbasin (complete)
- [x] put clean plate in diningtable (complete)

RECALLED HINTS:
- plate: countertop

RECALLED PROCEDURES: (none available — use `note` to record the procedure after completing the task)""",
        response="""<think>Task complete. Recalled procedures was empty, so I should note the clean workflow for future episodes.</think>
<action>note [clean task procedure: find object → take → go to sinkbasin → clean with sinkbasin → go to target receptacle → move]</action>""",
    ),
)
