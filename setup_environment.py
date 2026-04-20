#!/usr/bin/env python3
"""
Setup script for DS440 Project
Handles virtual environment creation, activation, and dependency installation
Run this script once on any device: python setup_environment.py
"""

import subprocess
import sys
import os
import platform

def run_command(command, shell=False):
    """Run a shell command and return success status"""
    try:
        subprocess.check_call(command, shell=shell)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to run command: {' '.join(command) if isinstance(command, list) else command}")
        print(f"        {e}")
        return False

def main():
    print("=" * 70)
    print("DS440 PROJECT SETUP")
    print("=" * 70)
    
    # Detect OS
    system = platform.system()
    is_windows = system == "Windows"
    
    venv_path = ".venv"
    
    # Step 1: Create virtual environment
    print("\n[1/6] Creating virtual environment...")
    if os.path.exists(venv_path):
        print("      Virtual environment already exists")
    else:
        if run_command([sys.executable, "-m", "venv", venv_path]):
            print("      Virtual environment created successfully")
        else:
            print("[ERROR] Failed to create virtual environment")
            return False
    
    # Step 2: Get pip executable path
    print("\n[2/6] Locating pip...")
    if is_windows:
        pip_executable = os.path.join(venv_path, "Scripts", "pip")
        python_executable = os.path.join(venv_path, "Scripts", "python")
    else:
        pip_executable = os.path.join(venv_path, "bin", "pip")
        python_executable = os.path.join(venv_path, "bin", "python")
    
    print("      Pip location found")
    
    # Step 3: Upgrade pip
    print("\n[3/6] Upgrading pip...")
    if run_command([pip_executable, "install", "--upgrade", "pip"]):
        print("      Pip upgraded successfully")
    else:
        print("[WARNING] Could not upgrade pip, continuing...")
    
    # Step 4: Install dependencies
    print("\n[4/6] Installing dependencies...")
    dependencies = [
        "pandas",
        "numpy",
        "scikit-learn",
        "dash",
        "plotly",
        "ipykernel",
        "jupyter",
        "matplotlib",
        "seaborn",
        "xgboost",
        "lightgbm",
        "catboost"
    ]
    
    if run_command([pip_executable, "install"] + dependencies):
        print(f"      Successfully installed {len(dependencies)} packages")
    else:
        print("[ERROR] Failed to install dependencies")
        return False
    
    # Step 5: Register kernel for Jupyter
    print("\n[5/6] Registering Jupyter kernel...")
    kernel_name = "ds440_env"
    if run_command([python_executable, "-m", "ipykernel", "install", "--user", "--name", kernel_name, "--display-name", "DS440 Environment"]):
        print("      Jupyter kernel registered successfully")
    else:
        print("[WARNING] Could not register Jupyter kernel, continuing...")
    
    # Step 6: Save instructions
    print("\n[6/6] Finalizing setup...")
    if is_windows:
        activate_cmd = f".\\{os.path.join(venv_path, 'Scripts', 'activate.bat')}"
    else:
        activate_cmd = f"source {os.path.join(venv_path, 'bin', 'activate')}"
    
    with open("SETUP_COMPLETE.txt", "w", encoding="utf-8") as f:
        f.write("DS440 PROJECT SETUP COMPLETE\n")
        f.write("=" * 70 + "\n\n")
        f.write("Virtual Environment Path: .venv\n")
        f.write(f"Python: {python_executable}\n")
        f.write(f"Pip: {pip_executable}\n\n")
        f.write("To activate the virtual environment:\n")
        f.write(f"    {activate_cmd}\n\n")
        f.write("To deactivate:\n")
        f.write("    deactivate\n\n")
        f.write("Note: Jupyter notebooks should automatically use the 'DS440 Environment' kernel.\n")
    
    print("      Setup complete")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print("\nYour project is now ready to use. All dependencies have been installed.")
    print("\nTo activate the virtual environment, run:")
    print(f"    {activate_cmd}")
    print("\nThen you can run Jupyter and start working on your notebooks.")
    print("\nFor detailed instructions, see SETUP_COMPLETE.txt")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
