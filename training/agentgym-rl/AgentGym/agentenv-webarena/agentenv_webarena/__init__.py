import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "webarena"),
)

# WebArena site URLs; export them to point at your own deployment.
os.environ.setdefault("SHOPPING", "http://127.0.0.1:7770")
os.environ.setdefault("SHOPPING_ADMIN", "http://127.0.0.1:7780/admin")
os.environ.setdefault("REDDIT", "http://127.0.0.1:9999")
os.environ.setdefault("GITLAB", "http://127.0.0.1:8023")
os.environ.setdefault("MAP", "http://127.0.0.1:3000")
os.environ.setdefault(
    "WIKIPEDIA",
    "http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
)
os.environ.setdefault("HOMEPAGE", "http://127.0.0.1:4399")

os.chdir("./webarena")

from .launch import launch
from .server import app
