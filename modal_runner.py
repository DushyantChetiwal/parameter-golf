import modal
import subprocess
import os
import glob
from datetime import datetime

app = modal.App("parameter-golf-multigpu")
data_volume = modal.Volume.from_name("fineweb-data")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "sentencepiece")
    # SURGICAL MOUNT: Bypasses .gitignore and live-syncs this exact file
    .add_local_file(
        local_path="records/track_10min_16mb/2026-03-27_Annealed_Muon_1.58bit/train_gpt.py",
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

    print("Launching torchrun across 2x A100s with persistent dataset attached...")
    print("Dynamo compiler logs are being routed to a file. Training steps will stream below:\n")
    print("-" * 50)
    
    # 1. We leave stdout alone (streams live to terminal)
    # 2. We route stderr (Dynamo spam + tracebacks) to a background file
    try:
        with open("dynamo_logs.txt", "w", encoding="utf-8") as err_file:
            subprocess.run([
                "torchrun", 
                "--standalone", 
                "--nproc_per_node=2", 
                "train_gpt.py" 
            ], env=env, check=True, stderr=err_file)
            
    except subprocess.CalledProcessError as e:
        # If it crashes, rip the last 40 lines of the hidden log file to show the Traceback!
        print("\n" + "❌" * 25)
        print("CRITICAL CRASH DETECTED! Pulling traceback from hidden logs:\n")
        if os.path.exists("dynamo_logs.txt"):
            with open("dynamo_logs.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
                print("".join(lines[-40:])) 
        print("❌" * 25)
        raise e

    print("-" * 50)
    print("\nTraining complete. Harvesting artifacts...")
    artifacts = {}
    
    if os.path.exists("final_model.int8.ptz"):
        with open("final_model.int8.ptz", "rb") as f:
            artifacts["model_ptz"] = f.read()
            
    if os.path.exists("dynamo_logs.txt"):
        with open("dynamo_logs.txt", "r", encoding="utf-8") as f:
            artifacts["dynamo_logs"] = f.read()
            
    log_files = glob.glob("logs/*.txt")
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        with open(latest_log, "r", encoding="utf-8") as f:
            artifacts["run_log"] = f.read()
            
    return artifacts

@app.local_entrypoint()
def main():
    print("Submitting training job to Modal...")
    artifacts = run_distributed.remote()
    
    print("\n--- RUN FINISHED. SAVING ARTIFACTS LOCALLY ---")
    
    if "dynamo_logs" in artifacts:
        os.makedirs("cloud_logs", exist_ok=True)
        log_path = f"cloud_logs/dynamo_logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(artifacts["dynamo_logs"])
        print(f"✅ Saved full Dynamo compiler logs to: ./{log_path}")

    if "run_log" in artifacts:
        os.makedirs("cloud_logs", exist_ok=True)
        log_path = f"cloud_logs/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(artifacts["run_log"])
        print(f"✅ Saved standard training logs to: ./{log_path}")
        
    if "model_ptz" in artifacts:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(artifacts["model_ptz"])
        print(f"✅ Saved quantized model to: ./final_model.int8.ptz ({len(artifacts['model_ptz']) / 1024 / 1024:.2f} MB)")