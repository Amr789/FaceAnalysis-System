import os
import shutil
import subprocess
import sys

def setup_kaggle_and_download():
    # 1. Check if kaggle.json is present in the current directory
    if not os.path.exists("kaggle.json"):
        print("❌ Error: 'kaggle.json' not found in the project root.")
        print("Please download your API token from Kaggle settings and place it here.")
        sys.exit(1)

    # 2. Setup directory and move token (Linux/Mac)
    # Note: On Windows, the config path is different (%HOMEPATH%/.kaggle/kaggle.json)
    # This script assumes a Linux/Mac/Cloud environment like Colab or standard servers.
    home = os.path.expanduser("~")
    kaggle_dir = os.path.join(home, ".kaggle")
    
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
        print(f"Created directory: {kaggle_dir}")

    # Copy file safely
    dest_file = os.path.join(kaggle_dir, "kaggle.json")
    shutil.copy("kaggle.json", dest_file)
    
    # Set permissions (Owner read/write only) - equivalent to chmod 600
    os.chmod(dest_file, 0o600)
    print("✅ kaggle.json installed and permissions set.")

    # 3. Download Dataset
    print("⬇️ Downloading UTKFace dataset...")
    try:
        # Using subprocess to run the kaggle command
        subprocess.run(["kaggle", "datasets", "download", "-d", "jangedoo/utkface-new", "--unzip", "-p", "data/"], check=True)
        print("✅ Download and extraction complete! Data is in 'data/' folder.")
    except subprocess.CalledProcessError:
        print("❌ Error during download. Make sure 'kaggle' is installed (pip install kaggle).")
    except FileNotFoundError:
        print("❌ 'kaggle' command not found. Please run 'pip install kaggle' first.")

if __name__ == "__main__":
    setup_kaggle_and_download()