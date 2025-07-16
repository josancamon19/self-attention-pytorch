cd /workspace
git clone https://github.com/josancamon19/transformers
cd /workspace/transformers/05-cs336/01/
pip install uv
apt update && apt install -y cmake
uv venv
source .venv/bin/activate
uv sync
sh lfs.sh
sh ../../github.sh
git lfs pull origin main