import os
from pathlib import Path
import glob

directory_path = Path('data/Experiment1')

for root, dirs, files in os.walk(directory_path):
    for txt_file in glob.glob(os.path.join(root, '*.txt')):
        print(txt_file)
        with open(txt_file, 'r') as file:
            content = file.read()
            print(content)
