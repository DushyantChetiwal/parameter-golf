import modal
import subprocess
import os
import glob
import re
from datetime import datetime

app = modal.App("parameter-golf-multigpu")
data_volume = modal.Volume.from_name("fineweb-data")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "sentencepiece", "zstandard")
    # SURGICAL MOUNT: Bypasses .gitignore and live-syncs this exact file
    .add_local_file(
        local_path="records/track_10min_16mb/2026-03-27_Annealed_Muon_1.58bit/ablation_C_8kv.py",
        remote_path="/root/project/train_gpt.py"
    )
)

@app.function(
    gpu="A100:2", 
    image=image, 
    volumes={"/cloud_data": data_volume}, 
    timeout=3600
)
def run_distributed():
    os.chdir("/root/project")
    
    env = os.environ.copy()
    env["DATA_PATH"] = "/cloud_data/data/datasets/fineweb10B_sp1024"
    env["TOKENIZER_PATH"] = "/cloud_data/data/tokenizers/fineweb_1024_bpe.model"
    env["PYTHONUNBUFFERED"] = "1"
    env["TORCH_LOGS"] = "+dynamo,recompiles,graph_breaks"
    env["MAX_WALLCLOCK_SECONDS"] = "600"

    print("Launching torchrun across 2x A100s with persistent dataset attached...")
    print("Dynamo compiler logs are being routed to a file. Training steps will stream below:\n")
    print("-" * 50)
    
    crashed = False
    with open("dynamo_logs.txt", "w", encoding="utf-8") as err_file:
        try:
            subprocess.run([
                "torchrun",
                "--standalone",
                "--nproc_per_node=2",
                "train_gpt.py"
            ], env=env, check=True, stderr=err_file)
        except subprocess.CalledProcessError:
            crashed = True

    if crashed:
        print("\n" + "=" * 50)
        print("CRITICAL CRASH DETECTED! Searching for Python traceback:\n")
        if os.path.exists("dynamo_logs.txt"):
            with open("dynamo_logs.txt", "r", encoding="utf-8") as f:
                content = f.read()
            tb_positions = [m.start() for m in re.finditer(r'Traceback \(most recent call last\)', content)]
            if len(tb_positions) >= 2:
                print(content[tb_positions[-2]:tb_positions[-2] + 4000])
            elif tb_positions:
                print(content[tb_positions[-1]:tb_positions[-1] + 4000])
            else:
                lines = content.splitlines()
                print("\n".join(lines[-60:]))
        print("=" * 50)

    print("-" * 50)
    status = "CRASHED -- harvesting artifacts" if crashed else "Training complete. Harvesting artifacts"
    print(f"\n{status}...")
    artifacts = {"crashed": crashed}

    for ptz_name in ("final_model.ptz", "final_model.int8.ptz"):
        if os.path.exists(ptz_name):
            with open(ptz_name, "rb") as f:
                artifacts["model_ptz"] = f.read()
                artifacts["model_ptz_name"] = ptz_name
            break

    if os.path.exists("dynamo_logs.txt"):
        with open("dynamo_logs.txt", "r", encoding="utf-8") as f:
            artifacts["dynamo_logs"] = f.read()

    log_files = glob.glob("logs/*.txt")
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        with open(latest_log, "r", encoding="utf-8") as f:
            artifacts["run_log"] = f.read()

    if os.path.exists("train_gpt.py"):
        with open("train_gpt.py", "r", encoding="utf-8") as f:
            artifacts["train_code"] = f.read()

    return artifacts

@app.local_entrypoint()
def main():
    print("Submitting training job to Modal...")
    artifacts = run_distributed.remote()

    crashed = artifacts.get("crashed", False)
    banner = "RUN CRASHED" if crashed else "RUN FINISHED"
    print(f"\n--- {banner}. SAVING ARTIFACTS LOCALLY ---")

    if "dynamo_logs" in artifacts:
        os.makedirs("cloud_logs", exist_ok=True)
        log_path = f"cloud_logs/dynamo_logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(artifacts["dynamo_logs"])
        print(f"Saved full Dynamo compiler logs to: ./{log_path}")

    if "run_log" in artifacts:
        os.makedirs("cloud_logs", exist_ok=True)
        log_path = f"cloud_logs/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(artifacts["run_log"])
        print(f"Saved standard training logs to: ./{log_path}")

    if "model_ptz" in artifacts:
        out_name = artifacts.get("model_ptz_name", "final_model.ptz")
        with open(out_name, "wb") as f:
            f.write(artifacts["model_ptz"])
        print(f"Saved quantized model to: ./{out_name} ({len(artifacts['model_ptz']) / 1024 / 1024:.2f} MB)")

    if "train_code" in artifacts:
        os.makedirs("cloud_logs", exist_ok=True)
        code_path = f"cloud_logs/code_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.py"
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(artifacts["train_code"])
        print(f"Saved training code snapshot to: ./{code_path}")

    if crashed:
        print("\n[CRASH] Check dynamo logs above for the actual Python traceback.")