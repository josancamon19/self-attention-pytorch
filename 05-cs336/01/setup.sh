cd /workspace
git clone https://github.com/josancamon19/transformers
cd /workspace/transformers/05-cs336/01/
python -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
ln -s pip3 .venv/bin/pip
which pip; which python
pip install uv
uv sync
sh lfs.sh
sh ../../github.sh
git lfs pull origin main