#!/usr/bin/env python3
"""Submit AML Singularity batch jobs for autobench experiments.

Each job:
  - Clones code from GitHub (new branch per job for traceability)
  - Copies data from mount to local disk
  - Runs one or more experiments sequentially
  - Copies results back to mount (date+config naming, no overwrites)
  - sleep infinity at the end (SSH in to debug if needed)

Usage:
    # Single config
    python submit_job.py --config configs/baseline_e011.json --dry-run

    # Multiple configs in one job (saves machine startup time)
    python submit_job.py --config configs/a.json configs/b.json configs/c.json

    # Override prompts/reps
    python submit_job.py --config configs/a.json --prompts 200 --reps 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.entities import JobResourceConfiguration, SshJobService
from azure.identity import DefaultAzureCredential

# --- Workspace ---
SUBSCRIPTION_ID = "b6dc87f3-c479-49c8-8cb5-7896da3ff895"
RESOURCE_GROUP = "AMLStudio"
WORKSPACE = "NewsFeedL2_AML"

# --- Virtual Cluster ---
VC_ARM_ID = (
    f"/subscriptions/{SUBSCRIPTION_ID}"
    "/resourceGroups/rg-cs-ranking-ml-singularity"
    "/providers/Microsoft.MachineLearningServices/virtualClusters/ranking"
)

# --- Managed Identity ---
UAI_RESOURCE_ID = (
    f"/subscriptions/{SUBSCRIPTION_ID}"
    f"/resourceGroups/{RESOURCE_GROUP}"
    "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"
)

SSH_PUB_KEY = (
    "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCpOz0QGUOBnEqMn+DwzbltVytWcFB/"
    "J10EpA0Rf5UXMtScYFKKYAi50qyhhdT5nj0LharII8p42w5MGPMLepqey6oFkVjDWrkT"
    "mzYe2nfkZpT9+GjGIEnbSvSL5CidsSWwDTzsgb5eLu0bExWHRwXscTLIfYQBNurdinw+"
    "z6k96DS1W4YTclJveoKFMJTT0ZpNd8FnGlQeJuO++xR1zVxK938rGEHO1bY3Aph3Pdg"
    "sTYliJvYNqihM/p+az8UK+zRNwRdbE175UZALbuD77mVuF8hG19ggLxi3HeyO9RE8t9V"
    "hNn6nyZDtMQtRxpgqx83tYSXqUatMwoHXiONQ1gMVbKhW6kNb7vvwCAOmUU/In4psgM"
    "1RiEv/VNVSV/9CusYDsCOvGPT0mOliaRMebA2KyHPmjkKdQNW8FTUM9No1cFsigMtsj"
    "84PwjcZYbPGFCTbifutUjav7p0PN+9AyLCOEyikX9SVGq06Qo4/oW5/aRMOQRwRala9S"
    "4pQjvj3kduQp8jNITMW+yn5AI6lgE457rbSmMpE6YxhVgVQjF1Mb6szxrQoMntqTW5O1"
    "ypr691vcnk9yph9fv9BVc+b+wdFbe8qHoYCDtDSBPYMq7GdwYqoDkpWdi5VBYRiMOGPN"
    "H5PB6S6xLBLD3Ybm+tHW/vvT6d/qjd3wYvg2dAuGwc/Mw== xinyizou@microsoft.com"
)

ENVIRONMENT = "azureml:vllm_gemma4:3"
INSTANCE_TYPE = "Singularity.ND12am_A100_v4"
DATASTORE_PATH = "azureml://datastores/adls_msn_dni_09_rankfun/paths/"

GIT_REPO = "https://github.com/xinyiZou/vllm-msn.git"
GIT_BRANCH = "zxy_dev_autobench"

HF_MODEL_REPO = "Xinyi0625/Gemma-4-26B-A4B-it-deploy"
LOCAL_MODEL_DIR = "/tmp/models/gemma4"

MOUNT_RESULTS_PATH = "shares/users/zxy/autobench/results"
MOUNT_DATA_PATH = "shares/users/zxy/autobench/data"


def get_ml_client() -> MLClient:
    return MLClient(
        DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE,
    )


def make_resource_config() -> JobResourceConfiguration:
    return JobResourceConfiguration(
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        properties={
            "singularity": {
                "slaTier": "Premium",
                "priority": "High",
                "enableAzmlInt": False,
                "locations": ["ukwest"],
            }
        },
    )


def build_job_command(
    configs: list[dict],
    prompts: int = 200,
    reps: int = 2,
    branch: str = GIT_BRANCH,
) -> str:
    """Build bash command that runs multiple experiments then sleeps."""
    job_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build per-experiment run blocks
    experiment_blocks = []
    for i, config in enumerate(configs, 1):
        config_json = json.dumps(config, indent=2)
        from config_space import config_summary
        summary = config_summary(config)
        # Unique result name: date_expN_summary-slug
        slug = summary.replace(" | ", "_").replace(" ", "")[:50]
        result_name = f"{job_ts}_exp{i}_{slug}"

        env_exports = []
        for key in ["VLLM_MOE_USE_DEEP_GEMM", "VLLM_USE_FUSED_MOE_GROUPED_TOPK",
                    "VLLM_FLASHINFER_MOE_BACKEND", "VLLM_HUMMING_MOE_GEMM_TYPE"]:
            if key in config:
                env_exports.append(f'export {key}="{config[key]}"')
        env_block = "\n".join(env_exports)

        block = f"""
echo ""
echo "{'='*60}"
echo "  Experiment {i}/{len(configs)}: {summary}"
echo "{'='*60}"
date

# Set MoE env vars for this experiment
{env_block}

cat > /tmp/config_{i}.json << 'CFGEOF'
{config_json}
CFGEOF

python run_experiment.py --config /tmp/config_{i}.json --prompts {prompts} --reps {reps} \\
    2>&1 | tee "/tmp/run_{i}.log" || echo "EXPERIMENT {i} FAILED (exit $?)"

# Copy results with unique names (never overwrite)
if [ -d run_results ]; then
    for f in run_results/*.json; do
        [ -f "$f" ] && cp "$f" "$RESULTS_DIR/{result_name}.json" 2>/dev/null || true
    done
    rm -rf run_results
fi
"""
        experiment_blocks.append(block)

    experiments_str = "\n".join(experiment_blocks)

    cmd = f"""#!/bin/bash
# Autobench Job — {job_ts} — {len(configs)} experiment(s)
# Branch: {branch}
# Machine stays alive via sleep infinity at end (SSH in to debug)

echo "=== Autobench Job Start ==="
echo "Timestamp: {job_ts}"
echo "Experiments: {len(configs)}"
date

# 1. Clone code on the experiment branch
git clone --branch {branch} --depth 1 {GIT_REPO} /tmp/vllm-msn
cd /tmp/vllm-msn/benchmarks/gemma4_moe_fp8/autobench

# 2. Copy data from mount to local disk (mount I/O is slow)
MOUNT=$AZURE_ML_INPUT_msndni
mkdir -p datasets
cp "$MOUNT/{MOUNT_DATA_PATH}/sc1_delta_v2.jsonl" datasets/
echo "Data copied to local disk ($(wc -l < datasets/sc1_delta_v2.jsonl) lines)"

# 3. Download model from HuggingFace (token injected via AML env vars)
echo "Downloading model from HuggingFace..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
huggingface-cli download {HF_MODEL_REPO} --local-dir {LOCAL_MODEL_DIR} --local-dir-use-symlinks False
echo "Model downloaded to {LOCAL_MODEL_DIR}"
ls {LOCAL_MODEL_DIR}/

# 4. Fixed env vars for A100
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export GEMMA4_TEXT_ONLY_MODEL_PATH="{LOCAL_MODEL_DIR}/text_only"
export GEMMA4_ASSISTANT_MODEL_PATH="{LOCAL_MODEL_DIR}/assistant"

# 5. Prepare results directory on mount
RESULTS_DIR="$MOUNT/{MOUNT_RESULTS_PATH}/{job_ts}"
mkdir -p "$RESULTS_DIR"

# 6. Run experiments sequentially
{experiments_str}

# 7. Copy aggregated results.tsv to mount (append to master + keep per-job copy)
if [ -f results.tsv ]; then
    cp results.tsv "$RESULTS_DIR/results.tsv"
    # Also append to master results file
    MASTER_TSV="$MOUNT/{MOUNT_RESULTS_PATH}/results_all.tsv"
    if [ -f "$MASTER_TSV" ]; then
        tail -n +2 results.tsv >> "$MASTER_TSV"
    else
        cp results.tsv "$MASTER_TSV"
    fi
fi

echo ""
echo "=== All experiments done ==="
echo "Results saved to: $RESULTS_DIR/"
echo "Branch: {branch}"
date
echo ""
echo "Machine staying alive (sleep infinity). SSH in to inspect or re-run."
echo "Cancel the job when done."

sleep infinity
"""
    return cmd


def submit_experiment(
    configs: list[dict],
    prompts: int = 200,
    reps: int = 2,
    branch: str = GIT_BRANCH,
    display_name: str | None = None,
    ml_client: MLClient | None = None,
    hf_token: str | None = None,
) -> str:
    """Submit one job running multiple experiments. Returns job name (ID)."""
    if ml_client is None:
        ml_client = get_ml_client()

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not set. Set it via environment variable or pass hf_token=."
        )

    from config_space import config_summary
    job_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = [config_summary(c) for c in configs]

    if display_name is None:
        if len(configs) == 1:
            display_name = f"autobench-{job_ts}-{summaries[0][:30]}"
        else:
            display_name = f"autobench-{job_ts}-{len(configs)}exps"

    job_cmd = build_job_command(configs, prompts=prompts, reps=reps, branch=branch)

    job = command(
        command=job_cmd,
        display_name=display_name,
        environment=ENVIRONMENT,
        compute=VC_ARM_ID,
        resources=make_resource_config(),
        inputs={
            "msndni": Input(
                type="uri_folder",
                path=DATASTORE_PATH,
                mode=InputOutputModes.RW_MOUNT,
            ),
        },
        environment_variables={
            "_AZUREML_SINGULARITY_JOB_UAI": UAI_RESOURCE_ID,
            "HF_TOKEN": hf_token,
        },
        services={
            "ssh": SshJobService(ssh_public_keys=SSH_PUB_KEY, nodes="all"),
        },
    )

    created = ml_client.jobs.create_or_update(job)
    print(f"  Job submitted: {created.name}")
    print(f"  Display name: {display_name}")
    print(f"  Branch: {branch}")
    print(f"  Experiments: {len(configs)}")
    for i, s in enumerate(summaries, 1):
        print(f"    {i}. {s}")
    print(f"  Studio: {created.studio_url}")
    return created.name


def poll_jobs(
    job_names: list[str],
    ml_client: MLClient | None = None,
    poll_interval: int = 60,
    timeout: int = 7200,
) -> dict[str, str]:
    """Poll until all jobs reach 'Running' or terminal state.

    Note: jobs end with sleep infinity, so they won't reach 'Completed'
    on their own. We poll until Running (experiments done, sleeping) or Failed.
    """
    if ml_client is None:
        ml_client = get_ml_client()

    # With sleep infinity, 'Running' means experiments are done (or running).
    # User manually cancels when satisfied.
    done_states = {"Completed", "Failed", "Canceled"}
    results: dict[str, str] = {}
    pending = set(job_names)
    start = time.time()

    while pending and (time.time() - start) < timeout:
        for name in list(pending):
            job = ml_client.jobs.get(name)
            status = job.status
            if status in done_states:
                results[name] = status
                pending.remove(name)
                print(f"  Job {name}: {status}")
        if pending:
            elapsed = int(time.time() - start)
            print(f"  Waiting... {len(pending)} jobs pending ({elapsed}s elapsed)")
            time.sleep(poll_interval)

    for name in pending:
        results[name] = "Timeout"
    return results


def main():
    ap = argparse.ArgumentParser(description="Submit AML autobench experiment job")
    ap.add_argument("--config", nargs="+", required=True,
                    help="Path(s) to config JSON file(s). Multiple = run in same job.")
    ap.add_argument("--prompts", type=int, default=200, help="Number of prompts")
    ap.add_argument("--reps", type=int, default=2, help="Repetitions per experiment")
    ap.add_argument("--branch", default=GIT_BRANCH,
                    help=f"Git branch to clone (default: {GIT_BRANCH})")
    ap.add_argument("--name", default=None, help="Custom job display name")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print job command without submitting")
    args = ap.parse_args()

    configs = []
    for cfg_path in args.config:
        with open(cfg_path) as f:
            configs.append(json.load(f))

    if args.dry_run:
        print("=== DRY RUN — Job Command ===\n")
        print(build_job_command(configs, prompts=args.prompts, reps=args.reps,
                                branch=args.branch))
        print(f"\n=== Would submit {len(configs)} experiment(s) on branch '{args.branch}' ===")
        return 0

    job_name = submit_experiment(
        configs,
        prompts=args.prompts,
        reps=args.reps,
        branch=args.branch,
        display_name=args.name,
    )
    print(f"\nJob submitted: {job_name}")
    print("Job will sleep infinity after experiments. SSH in or cancel when done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
