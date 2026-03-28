import modal
import subprocess
import os
import glob

app = modal.App("parameter-golf-multigpu")
data_volume = modal.Volume.from_name("fineweb-data")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "sentencepiece")
    .add_local_dir(".", remote_path="/root/project")
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

    print("Launching torchrun across 2x A100s with persistent dataset attached...")
    subprocess.run([
        "torchrun", 
        "--standalone", 
        "--nproc_per_node=2", 
        "records/track_10min_16mb/2026-03-27_Annealed_Muon_1.58bit/train_gpt.py"
    ], env=env, check=True)

    # --- HARVEST PHASE: Grab the files before the container dies ---
    print("Training complete. Harvesting artifacts...")
    artifacts = {}
    
    # 1. Grab the compressed 16MB model
    if os.path.exists("final_model.int8.ptz"):
        with open("final_model.int8.ptz", "rb") as f:
            artifacts["model_ptz"] = f.read()
            
    # 2. Grab the newest log file
    log_files = glob.glob("logs/*.txt")
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        with open(latest_log, "r", encoding="utf-8") as f:
            artifacts["run_log"] = f.read()
            
    # Send the raw bytes back to your local laptop
    return artifacts

@app.local_entrypoint()
def main():
    print("Submitting training job to Modal...")
    
    # This will block and stream the stdout until the run finishes,
    # then it receives the returned artifacts dict!
    artifacts = run_distributed.remote()
    
    print("\n--- RUN FINISHED. SAVING ARTIFACTS LOCALLY ---")
    
    if "run_log" in artifacts:
        os.makedirs("cloud_logs", exist_ok=True)
        with open("cloud_logs/latest_run.txt", "w", encoding="utf-8") as f:
            f.write(artifacts["run_log"])
        print("✅ Saved logs to: ./cloud_logs/latest_run.txt")
        
    if "model_ptz" in artifacts:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(artifacts["model_ptz"])
        print(f"✅ Saved quantized model to: ./final_model.int8.ptz ({len(artifacts['model_ptz']) / 1024 / 1024:.2f} MB)")