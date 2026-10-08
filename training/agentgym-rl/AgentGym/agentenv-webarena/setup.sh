#!/bin/bash
cd ./webarena
pip install -r requirements.txt
playwright install-deps
playwright install 
pip install -e .
pip3 install gunicorn

export SHOPPING="${SHOPPING:-http://127.0.0.1:7770}"
export SHOPPING_ADMIN="${SHOPPING_ADMIN:-http://127.0.0.1:7780/admin}"
export REDDIT="${REDDIT:-http://127.0.0.1:9999}"
export GITLAB="${GITLAB:-http://127.0.0.1:8023}"
export MAP="${MAP:-http://127.0.0.1:3000}"
export WIKIPEDIA="${WIKIPEDIA:-http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing}"
export HOMEPAGE="${HOMEPAGE:-http://127.0.0.1:4399}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"


python scripts/generate_test_data.py
mkdir -p ./.auth
python browser_env/auto_login.py
python agent/prompts/to_json.py

cd ..

pip install -e .
