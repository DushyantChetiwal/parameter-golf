import modal
import subprocess
import os
import shutil

app = modal.App("parameter-golf-data-prep")

# 1. Grab your empty cloud hard drive
data_volume = modal.Volume.from_name("fineweb-data")

# 2. FIX: The Image now natively mounts your local code directory!
image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub")
    .add_local_dir(".", remote_path="/root/project")
)

@app.function(
    image=image, 
    volumes={"/cloud_data": data_volume}, 
    timeout=3600  # 1 hour timeout for downloading
)
def run_prep():
    print("Setting up cloud directory structure...")
    
    cloud_data_dir = "/cloud_data/data"
    os.makedirs(cloud_data_dir, exist_ok=True)
    
    # Copy the script INTO the volume
    source_script = "/root/project/data/cached_challenge_fineweb.py"
    target_script = os.path.join(cloud_data_dir, "cached_challenge_fineweb.py")
    shutil.copy(source_script, target_script)
    
    os.chdir(cloud_data_dir)
    print("Starting datacenter-speed download directly to Modal Volume...")
    
    subprocess.run([
        "python", 
        "cached_challenge_fineweb.py",
        "--variant", "sp1024"
    ], check=True)
    
    data_volume.commit()
    print("Data successfully downloaded and permanently saved to the cloud hard drive!")

@app.local_entrypoint()
def main():
    run_prep.remote()