import subprocess
import time
import os

start = time.time()
scripts_directory = '../trainer'
# ディレクトリ内のすべてのPythonファイルを取得
python_scripts = [f for f in os.listdir(scripts_directory) if f.endswith('.py')]
python_scripts.sort()
print(python_scripts)
# 各スクリプトを順に実行
for script in python_scripts:
    script_path = os.path.join(scripts_directory, script)
    print(f"Running script: {script_path}")
    # `caffeinate` コマンドを使用してスクリプトを実行
    subprocess.run(["caffeinate", "-i", "python", script_path])

print("All scripts have been executed.")
end = time.time()
print(f"Total execution time: {end - start} seconds")