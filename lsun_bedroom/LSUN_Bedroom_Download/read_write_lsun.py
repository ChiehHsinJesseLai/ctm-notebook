import argparse
import io
import os
import sys

from PIL import Image
import lmdb
import numpy as np

def dump_images(out_dir, images, prefix):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open('logfile.txt', 'w') as file:
        for i, img in enumerate(images):
            if i < 2835243:
                file.write(str(i)+'\t')
                file.flush()
                continue
            file.write("creating :"+str(i)+'\t')
            file.flush()
            Image.fromarray(img).save(os.path.join(out_dir, f"bedroom_{prefix}_{i:07d}.png"))

def read_images(lmdb_path):
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_readers=2, readonly=True)
    
    with env.begin(write=False) as transaction:
        cursor = transaction.cursor()
        for _, webp_data in cursor:
            try:
                img = Image.open(io.BytesIO(webp_data))
            except Exception as e:
                pass
            arr = np.array(img)
            yield arr
print("start")

train_lmdb_path = "/dataset/LSUN_Bedroom/bedroom_train_lmdb"
val_lmbd_path = "/dataset/LSUN_Bedroom/bedroom_val_lmdb"
train_out_dir = "/dataset/LSUN_Bedroom/train_orig_imgs"
val_out_dir = "/dataset/LSUN_Bedroom/val_orig_imgs"

print("read & demp the train dataset")
images = read_images(train_lmdb_path)
dump_images(train_out_dir, images, "train")
print("read & demp the validation dataset")
images = read_images(val_lmbd_path)
dump_images(val_out_dir, images, "val")
print("done")