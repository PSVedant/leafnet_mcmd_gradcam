import os
import shutil

BASE = "."

RAW = os.path.join(BASE, "data", "raw")
TRAIN_RAW = os.path.join(RAW, "train")
VAL_RAW = os.path.join(RAW, "val")
TEST_RAW = os.path.join(RAW, "test")

os.makedirs(TRAIN_RAW, exist_ok=True)
os.makedirs(VAL_RAW, exist_ok=True)
os.makedirs(TEST_RAW, exist_ok=True)

# -------- Your existing paths -------- #
TRAIN_SRC = os.path.join(BASE, "New Plant Diseases Dataset(Augmented)", 
                         "New Plant Diseases Dataset(Augmented)", "train")

VAL_SRC = os.path.join(BASE, "New Plant Diseases Dataset(Augmented)", 
                       "New Plant Diseases Dataset(Augmented)", "valid")

TEST_SRC = os.path.join(BASE, "test", "test")

# -------- Copy Train -------- #
shutil.copytree(TRAIN_SRC, TRAIN_RAW, dirs_exist_ok=True)

# -------- Copy Val -------- #
shutil.copytree(VAL_SRC, VAL_RAW, dirs_exist_ok=True)

# -------- Copy Test Images -------- #
for img in os.listdir(TEST_SRC):
    shutil.copy(os.path.join(TEST_SRC, img), TEST_RAW)

print("Raw dataset created successfully inside data/raw/")
