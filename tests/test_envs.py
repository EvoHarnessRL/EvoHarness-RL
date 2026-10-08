from evoharness.envs import get_domain
from evoharness.envs.alfworld.belief import AlfWorldBelief
from evoharness.envs.webarena.belief import WebArenaBelief
from evoharness.envs.webarena.env import WebArenaEnv, extract_objective, load_task_ids
from evoharness.envs.webshop.belief import WebShopBelief

ALL = {"belief", "plan", "experience"}


def test_alfworld_belief_tracks_objects_and_locations():
    belief = AlfWorldBelief()
    belief.reset("put a clean egg in fridge", "")
    belief.update("go to countertop 1", "You arrive at countertop 1. On the countertop 1, you see a egg 1, and a knife 2.", [], False)
    progress = belief.update("take egg 1 from countertop 1", "You pick up the egg 1 from the countertop 1.", [], False)
    assert progress["status"] == "partial"
    tracked = belief.track("egg")
    assert "egg 1 (target): isPickedUp=True" in tracked and "egg 1 held_by agent" in tracked
    assert "knife 2 on countertop 1" in belief.render(["egg"]) or "knife 2" in belief.render([])
    assert belief.update("go to sink 1", "Nothing happens.", [], False)["status"] == "blocked"
    assert "Visited locations: countertop 1" in belief.track("apple")


def test_webshop_belief_parses_results_and_item_pages():
    belief = WebShopBelief()
    belief.reset("i want red shoes, and price lower than 50.00 dollars", "")
    assert belief.ceiling == 50.0 and "red" in belief.target_attributes
    results = "'Back to Search' [SEP] 'Page 1 (Total results: 2)' [SEP] 'B078GWRC1J' [SEP] 'Red Shoe' [SEP] '$39.99' [SEP] 'B000000001' [SEP] 'Blue Shoe' [SEP] '$79.00'"
    belief.update("search[red shoes]", results, [], False)
    candidates = belief.track("candidates")
    assert "B078GWRC1J: $39.99 [under budget] Red Shoe" in candidates and "[over budget]" in candidates
    item = "'Back to Search' [SEP] 'Red Shoe' [SEP] 'Price: $39.99' [SEP] 'size' [SEP] '8' [SEP] 'Buy Now'"
    belief.update("click[b078gwrc1j]", item, [], False)
    belief.update("click[8]", item, [], False)
    assert "within budget" in belief.track("price")
    assert "selected so far: 8" in belief.track("options")
    assert "B078GWRC1J" in belief.track("red shoe")


FRAME = "[1] RootWebArea 'Orders' focused: True\n[5] StaticText 'Grand Total $120.50'\n[6] link 'Shipping'\nURL: http://shop/orders\nOBJECTIVE: What is the grand total of order 5?"


def test_webarena_belief_named_tracks():
    belief = WebArenaBelief()
    belief.reset("What is the grand total of order 5?", FRAME)
    belief.update("click [6]", FRAME.replace("Orders", "Shipping").replace("/orders", "/ship"), [], False)
    assert "[step 0] Orders — http://shop/orders" in belief.track("visited")
    assert "Grand Total $120.50" in belief.track("values")
    assert "grand total" in belief.track("objective").lower()
    assert "Shipping" in belief.track("shipping")
    assert "current page: Shipping — http://shop/ship" in belief.render([])


def test_webarena_env_with_fake_server():
    class Server:
        def __init__(self):
            self.sent = []

        def reset(self, index):
            self.index = index

        def observe(self):
            return FRAME + "\nPREVIOUS ACTION: None"

        def step(self, action):
            self.sent.append(action)
            if action == "```click [99]```":
                return {"observation": "Element 99 not found", "reward": 0, "terminated": False, "info": None}
            done = action.startswith("```stop")
            return {"observation": FRAME, "reward": 1.0, "terminated": done, "truncated": False, "info": {}}

        def close(self):
            pass

    env = WebArenaEnv("http://unused")
    env.server = Server()
    start = env.reset(get_domain("webarena").list_tasks("all", start=3, end=4)[0])
    assert start.objective == "What is the grand total of order 5?" and "PREVIOUS ACTION" not in start.observation
    assert not env.step("dance").valid
    failed = env.step("click [99]")
    assert not failed.valid and "not found" not in failed.observation
    assert not env.step("click [6]").done
    final = env.step("stop [$120.50]")
    assert final.won and env.server.sent[-1] == "```stop [$120.50]```"
    assert extract_objective(FRAME) == "What is the grand total of order 5?"

    env.server.step = lambda a: {"observation": "", "reward": 0.0, "terminated": False, "truncated": True, "info": {}}
    assert env.step("scroll [down]").done


def test_webarena_task_files(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text('{"task_ids": ["webarena_24", "webarena_7"]}')
    assert load_task_ids(str(path), 0, 0) == [24, 7]
    assert load_task_ids(None, 2, 5) == [2, 3, 4]


def test_prompts_advertise_only_enabled_modules():
    for name in ("alfworld", "webshop", "webarena"):
        domain = get_domain(name)
        full = domain.system_prompt("inline", ALL)
        assert "track [" in full and "recall [" in full and "commit [" in full
        no_belief = domain.system_prompt("inline", ALL - {"belief"})
        assert "track [" not in no_belief and "commit [" in no_belief
        assert "commit [" not in domain.system_prompt("always_on", ALL)
        assert "commit [" not in domain.system_prompt("env_only", set())


def test_task_type_detection():
    assert get_domain("alfworld").task_type("put a hot apple in fridge") == "pick_heat_then_place_in_recep"
    assert get_domain("alfworld").task_type("put two cds in safe") == "pick_two_obj_and_place"
    assert get_domain("webshop").task_type("i need a usb charger for my laptop") == "electronics"
    assert get_domain("webarena").task_type("Get directions and driving time from CMU") == "map_directions"


def test_alfworld_search_priorities_strip_instance_numbers():
    experience = get_domain("alfworld").experience
    found = experience.search_priorities("x", ["go to fridge 1", "take egg 2 from fridge 1"])
    assert found == {"egg": ["fridge"]}
    assert experience.generalize("take egg 2 from fridge 1") == "take egg from fridge"
