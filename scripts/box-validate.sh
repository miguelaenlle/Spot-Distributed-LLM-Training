#!/usr/bin/env bash
# Validate the training path on a provisioned box, in isolation from the
# orchestrator. Run it over SSH after provisioning to confirm the interpreter +
# env + trainer all work before relying on the full user-data boot path.
#
#   ssh ubuntu@<ip>  # then:
#   bash ~/app/scripts/box-validate.sh          # 60s smoke run (override MAX_SECONDS)
set -x

# Interpreter: the DLAMI base shell only has `python3`; the pytorch venv python
# (which has torch + boto3) lives here and exists even when the login shell
# doesn't auto-activate. Fall back to python3 on other AMIs.
VENV_PY=/opt/pytorch/bin/python
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"
echo "INTERPRETER: $VENV_PY"

"$VENV_PY" -c "import sys,torch; print('exe',sys.executable,'torch',torch.__version__,'cuda',torch.cuda.is_available())" || exit 1
"$VENV_PY" -c "import boto3; print('boto3 ok')" || "$VENV_PY" -m pip install boto3 || exit 1

cd ~/app || { echo "no ~/app — provisioning never cloned"; exit 1; }
source ~/spot-train.env
MAX_SECONDS=${MAX_SECONDS:-60} "$VENV_PY" -u -m spot_train.train
echo "box-validate exit rc=$?"
