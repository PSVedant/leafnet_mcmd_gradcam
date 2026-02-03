import os, shutil, random

RAW_TRAIN = "data/raw/train"
RAW_VAL = "data/raw/val"

FINAL_TRAIN = "data/train"
FINAL_VAL = "data/val"
FINAL_TEST = "data/test"

for p in [FINAL_TRAIN, FINAL_VAL, FINAL_TEST]:
    os.makedirs(p, exist_ok=True)

print("Merging raw train + val...")

temp = "data/temp"
os.makedirs(temp, exist_ok=True)

for cls in os.listdir(RAW_TRAIN):
    os.makedirs(os.path.join(temp, cls), exist_ok=True)

    for folder in [RAW_TRAIN, RAW_VAL]:
        p = os.path.join(folder, cls)
        if os.path.exists(p):
            for img in os.listdir(p):
                shutil.copy(os.path.join(p, img), os.path.join(temp, cls))

print("Splitting into train/val/test...")

for cls in os.listdir(temp):
    imgs = os.listdir(os.path.join(temp, cls))
    random.shuffle(imgs)

    n = len(imgs)
    t = int(0.7*n)
    v = int(0.85*n)

    for i, img in enumerate(imgs):
        src = os.path.join(temp, cls, img)

        if i < t:
            dst = os.path.join(FINAL_TRAIN, cls)
        elif i < v:
            dst = os.path.join(FINAL_VAL, cls)
        else:
            dst = os.path.join(FINAL_TEST, cls)

        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, os.path.join(dst, img))

print("DONE ✅")
