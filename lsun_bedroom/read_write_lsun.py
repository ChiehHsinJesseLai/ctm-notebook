import argparse
import io
import os

from PIL import Image
import lmdb
import numpy as np

def dump_images(out_dir, images, prefix):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(out_dir, f"bedroom_{prefix}_{i:07d}.png"))

def read_images(lmdb_path):
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_readers=2, readonly=True)
    with env.begin(write=False) as transaction:
        cursor = transaction.cursor()
        for _, webp_data in cursor:
            try:
                img = Image.open(io.BytesIO(webp_data))
            except:
                pass
            arr = np.array(img)
            yield arr

train_lmdb_path = "/data2/LSUN_Bedroom/bedroom_train_lmdb"
val_lmbd_path = "/data2/LSUN_Bedroom/bedroom_val_lmdb"
train_out_dir = "/data2/LSUN_Bedroom/train_orig_imgs"
val_out_dir = "/data2/LSUN_Bedroom/val_orig_imgs"

images = read_images(train_lmdb_path)
dump_images(train_out_dir, images, "train")

images = read_images(val_lmbd_path)
dump_images(val_out_dir, images, "val")