import subprocess
import time
import os

start = time.time()
scripts_directory = r'..\trainer'
# Get all Python files in the directory
python_scripts = [f for f in os.listdir(scripts_directory) if f.endswith('.py')]
python_scripts.sort()

for script in python_scripts:
    script_path = os.path.join(scripts_directory, script)
    print(f"Running script: {script_path}")
    subprocess.run(["caffeinate", "-i", "python", script_path])
    # for windows
    #subprocess.run(["python", script_path], check=True)

print("All scripts have been executed.")
end = time.time()
print(f"Total execution time: {end - start} seconds")