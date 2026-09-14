import modal
import os
import subprocess

requirements_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'requirements.txt'))
image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements(requirements_path)

app = modal.App("tsfm-training", image=image)

# Mount the entire wam directory so the remote has access to everything
wam_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mount = modal.Mount.from_local_dir(wam_dir, remote_path="/root/wam")

def get_file_states(base_dir):
    """Helper to record modification times of files to detect changes."""
    states = {}
    for root, _, files in os.walk(base_dir):
        if '.git' in root or '__pycache__' in root:
            continue
        for f in files:
            path = os.path.join(root, f)
            states[path] = os.path.getmtime(path)
    return states

@app.function(
    gpu="any", # You can specify e.g. gpu="A10G" or gpu="A100" if you need more power
    mounts=[mount],
    timeout=86400, # 24 hours
)
def run_experiments():
    import sys
    sys.path.append("/root/wam")
    
    from tsfm.experiments_unified import main
    
    target_dir = "/root/wam/tsfm"
    os.chdir(target_dir)
    
    # Record file state before running
    before_states = get_file_states(target_dir)
    
    # Run the main training and evaluation function
    print("Running experiments on Modal GPU...")
    main()
    print("Experiments finished on Modal GPU.")
    
    # Identify generated/modified files
    after_states = get_file_states(target_dir)
    changed_files = {}
    
    for path, mtime in after_states.items():
        if path not in before_states or mtime > before_states[path]:
            with open(path, "rb") as f:
                # Store relative path and binary contents
                rel_path = os.path.relpath(path, target_dir)
                changed_files[rel_path] = f.read()
                print(f"Captured modified file: {rel_path}")
                
    return changed_files

@app.local_entrypoint()
def main():
    print("Submitting training job to Modal...")
    changed_files = run_experiments.remote()
    
    if not changed_files:
        print("No files were modified or generated on Modal.")
        return
        
    print(f"Received {len(changed_files)} modified files from Modal. Saving locally...")
    
    target_dir = os.path.dirname(__file__)
    for rel_path, contents in changed_files.items():
        local_path = os.path.join(target_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(contents)
        print(f"Saved {rel_path}")
        
    print("\n--- Pushing to GitHub (Locally) ---")
    repo_dir = os.path.abspath(os.path.join(target_dir, '..'))
    try:
        subprocess.run(['git', 'add', '.'], cwd=repo_dir, check=True)
        # Check if there are actually changes to commit
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=repo_dir, capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(['git', 'commit', '-m', 'Update foundation model and evaluation plots from Modal run'], cwd=repo_dir, check=True)
            subprocess.run(['git', 'push'], cwd=repo_dir, check=True)
            print("Successfully pushed to GitHub!")
        else:
            print("No changes to commit to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to GitHub. Error: {e}")
