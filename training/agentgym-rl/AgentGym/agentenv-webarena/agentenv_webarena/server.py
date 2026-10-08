"""
Async Flask (Quart) Server
"""

from quart import Quart, jsonify, request
from quart_cors import cors
import subprocess
import json
import glob
import os
import sys
import time
import datetime
import asyncio
import requests
from .environment import webarena_env_server

app = Quart(__name__)

VISUAL = os.environ.get("VISUAL", "false").lower() == "true"
if VISUAL:
    print("Running in VISUAL mode")
    app = cors(
        app, 
        allow_origin="*",
    )
_max_id=0
_max_id_lock=asyncio.Lock()

_last_auth_probe = 0.0
_AUTH_PROBE_EVERY = 900.0

@app.route("/", methods=["GET"])
async def generate_ok():
    """Test connectivity"""
    return "ok"

# WebArena site URLs are exported into the process env by agentenv_webarena/__init__.py.
_SITE_ENV_VARS = [
    "SHOPPING",
    "SHOPPING_ADMIN",
    "REDDIT",
    "GITLAB",
    "MAP",
    "WIKIPEDIA",
    "HOMEPAGE",
]


def _probe_site(url: str, timeout: float = 5.0) -> dict:
    """Blocking single-site probe. Any HTTP response < 500 means the server is up
    (redirects / auth 4xx still indicate a live site); >=500 or an exception is
    treated as down."""
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True)
        ok = r.status_code < 500
        return {"ok": ok, "status": r.status_code}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@app.route("/health", methods=["GET"])
async def health():
    """Probe all configured WebArena sites and report per-site status plus an
    overall ``ready`` boolean, so clients can gate startup on site health."""
    sites = {name: os.environ.get(name) for name in _SITE_ENV_VARS}
    sites = {name: url for name, url in sites.items() if url}

    async def probe(name, url):
        return name, await asyncio.to_thread(_probe_site, url)

    results = await asyncio.gather(*(probe(n, u) for n, u in sites.items()))
    per_site = {name: res for name, res in results}
    ready = bool(per_site) and all(v.get("ok") for v in per_site.values())
    return jsonify({"ready": ready, "sites": per_site})

@app.route("/list_envs", methods=["GET"])
async def list_envs():
    """List all environments. """
    return jsonify(list(webarena_env_server.env.keys()))

@app.route("/create", methods=["POST"])
async def create():
    """Create a new environment"""
    global _max_id
    async with _max_id_lock:
        env_idx = _max_id
        _max_id += 1
    env = await asyncio.to_thread(webarena_env_server.create, env_idx)
    return jsonify({"env_idx": env})

@app.route("/step", methods=["POST"])
async def step():
    """
    Make an action
    """
    step_query = await request.get_json()
    step_data = await asyncio.to_thread(
        webarena_env_server.step, step_query["env_idx"], step_query["action"]
    )
    step_response = {
        "observation": step_data[0],
        "reward": step_data[1],
        "terminated": step_data[2],
        "truncated": step_data[3],
        "info": step_data[4],
    }
    return jsonify(step_response)

@app.route("/observation", methods=["GET"])
async def get_observation():
    """
    current observation
    """
    env_idx = request.args.get("env_idx", type=int)
    obs = await asyncio.to_thread(webarena_env_server.observation, env_idx)
    return jsonify(obs)

@app.route("/observation_metadata", methods=["GET"])
async def get_obsmetadata():
    """
    current observation metadata
    """
    env_idx = request.args.get("env_idx", type=int)
    obs_meta = await asyncio.to_thread(webarena_env_server.observation_metadata, env_idx)
    return jsonify(obs_meta)

def _session_is_dead(state_file: str, url: str, signed_out_marker: str, timeout: float = 20.0) -> bool:
    """True when the cookies in state_file no longer authenticate against url.

    GitLab and Magento invalidate sessions SERVER-side while the cookie itself keeps a
    far-future (or session-scoped) expiry, and every WebArena site serves its login wall
    with HTTP 200. So neither the declared expiry nor the status code can detect a logged
    out session -- only the page content can.
    """
    try:
        with open(state_file) as f:
            data = json.load(f)
    except Exception:
        return True
    jar = {c["name"]: c["value"] for c in data.get("cookies", [])}
    if not jar:
        return True
    try:
        r = requests.get(url, cookies=jar, timeout=timeout, allow_redirects=True)
    except Exception:
        return False  # site unreachable is a different failure; do not thrash re-login
    return signed_out_marker in r.text


# state file, authenticated URL, marker that only appears when signed OUT
_AUTH_CHECKS = [
    ("gitlab_state.json", "{}/dashboard/projects", "Sign in"),
    ("shopping_state.json", "{}/customer/account/", "Customer Login"),
    ("shopping_admin_state.json", "{}/dashboard", "please sign in"),
    ("reddit_state.json", "{}/", "Log in"),
]
_AUTH_BASE_ENV = {
    "gitlab_state.json": "GITLAB",
    "shopping_state.json": "SHOPPING",
    "shopping_admin_state.json": "SHOPPING_ADMIN",
    "reddit_state.json": "REDDIT",
}


def check_cookies_expiration():
    relogin_tolerance = 2700 # 1 hours in seconds
    now = datetime.datetime.now().timestamp()
    earliest_expiry = float('inf')
    auth_dir = ".auth"
    cookie_files = glob.glob(os.path.join(auth_dir, "*.json"))
    if not cookie_files:
        return True
    for cookie_file in cookie_files:
        try:
            with open(cookie_file, 'r') as f:
                data = json.load(f)
            if not data.get('cookies'):
                continue
            for cookie in data['cookies']:
                if 'expires' in cookie:
                    expires = float(cookie['expires'])
                    if expires <= 0:
                        continue
                    earliest_expiry = min(earliest_expiry, expires)
        except Exception as e:
            print(f"Reading {cookie_file} Error: {e}")
    if earliest_expiry != float('inf'):
        time_to_expiry = earliest_expiry - now
        # print(f"oldest cookie will be expired in {time_to_expiry/3600:.2f} hours")
        if time_to_expiry < relogin_tolerance:
            return True

    # The declared expiry said we are fine, which it also does for a session that died
    # server-side. Confirm with a real authenticated request, rate-limited so a multi-day
    # run does not pay for it on every reset.
    global _last_auth_probe
    if now - _last_auth_probe < _AUTH_PROBE_EVERY:
        return False
    _last_auth_probe = now
    for state_file, url_tmpl, marker in _AUTH_CHECKS:
        path = os.path.join(auth_dir, state_file)
        base = os.environ.get(_AUTH_BASE_ENV[state_file])
        if not base or not os.path.exists(path):
            continue
        if _session_is_dead(path, url_tmpl.format(base.rstrip("/")), marker):
            print(f"[auth] {state_file} no longer authenticates -> re-login")
            return True
    return False

@app.route("/reset", methods=["POST"])
async def reset():
    """
    reset the environment
    """
    reset_query = await request.get_json()
    reset_query["options"] = {
        "config_file": f"./config_files/{reset_query['idx']}.json"
    }
    try:
        if check_cookies_expiration():
            print("cookie will be expired in one hour, executeing auto_login...")
            # sys.executable, not "python": this server is launched by absolute path from
            # the mmevo env without that env on PATH, so a bare "python" resolves to a
            # different interpreter that has no playwright. Check the result too -- a
            # silently failing re-login is what lets a run train against login walls.
            proc = subprocess.run(
                [sys.executable, "browser_env/auto_login.py"],
                capture_output=True,
                text=True,
            )  # This will take time.
            if proc.returncode != 0:
                print(f"[auth] auto_login FAILED rc={proc.returncode}: {proc.stderr[-2000:]}")
            else:
                print("[auth] auto_login completed")
            
        obs, info, sites,object = await asyncio.to_thread(
            webarena_env_server.reset,
            reset_query["env_idx"], reset_query["seed"], reset_query["options"]
        )
    except Exception as e:
        print(e)
        obs={"text": "TimeoutError"}
        sites = "Error"
        object = "Error"
    reset_response = {"observation": obs["text"], "sites": sites, "object": object}
    return jsonify(reset_response)

@app.route("/close", methods=["POST"])
async def close():
    close_query = await request.get_json()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(webarena_env_server.close, close_query["env_idx"]),
            timeout=30.0
        )
        close_response = {"closed":"closed"}
    except asyncio.TimeoutError:
        print(f"Close Env {close_query['env_idx']} time out.")
        close_response = {"closed":f"Close Env {close_query['env_idx']} time out."}
    except Exception as e:
        close_response = {"closed":f"Close Env {close_query['env_idx']} Error: {e}"}
    return jsonify(close_response)

def handle_exit(signum, frame):
    print("\n[INFO] Shutting down server gracefully...")
    for env_idx in list(webarena_env_server.envs.keys()):
        print(f"Closing environment {env_idx}...")
        try:
            webarena_env_server.close(env_idx)
        except Exception as e:
            print(f"Error closing environment {env_idx}: {e}")
    try:
        auth_dir = "./.auth"
        print(f"[INFO] Cleaning up {auth_dir} folder...")
        if os.path.exists(auth_dir) and os.path.isdir(auth_dir):
            auth_files = glob.glob(os.path.join(auth_dir, "*.json"))
            for file_path in auth_files:
                try:
                    os.remove(file_path)
                    print(f"  Removed {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"  Error removing {os.path.basename(file_path)}: {e}")
            print(f"[INFO] Removed {len(auth_files)} auth files")
        else:
            print(f"[INFO] Auth directory {auth_dir} not found or not a directory")
    except Exception as e:
        print(f"[ERROR] Failed to clean up auth folder: {e}")

    print("[INFO] Cleanup complete. Exiting now.")
    os._exit(0)

# 捕获 SIGINT（CTRL+C）
import signal
signal.signal(signal.SIGINT, handle_exit)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)