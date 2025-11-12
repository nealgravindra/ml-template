!/bin/bash

set -e

# 1) Get project into the VM by running in colab:
# !git clone https://github.com/nealgravindra/ml-template.git
# %cd ml-template

# run the following bash script
# 2) install / update uv (usually already part of colab)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# 3) Enforce exact environment for this python version / interpreter
uv export --format requirements-txt > requirements.lock.txt

# 4) make the *current kernel* match the lock
uv pip sync --system requirements.lock.txt

# 5) install your project into current kernel *without changing deps*
uv pip install --system -e . --no-deps

# 6) in colab, can run:
# !python -m ml_template.train --config configs/default.yaml
