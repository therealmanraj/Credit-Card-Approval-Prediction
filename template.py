import os
import logging
from pathlib import Path

logging.basicConfig(filename="CCAP.logs", level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_of_files = [
    ".github.com/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/utils/__init__.py",
    f"src/utils/common.py",
    f"src/logging/__init__.py",
    f"src/pipeline/__init__.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "notebooks/CCFD.ipynb",
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Created directory: {file_dir} for the file: {file_name}")
        
    if ((not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0)):
        with open(file_path,'w'):
            pass
        logging.info(f"{file_path} file created")
    else:
        logging.info(f"File {file_path} already exists and has content.")