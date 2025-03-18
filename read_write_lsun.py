""" 
to read and write lsun bedroom images
"""
import io
import os
from PIL import Image
import lmdb
import numpy as np

def dump_images(out_dir, images, prefix):
    """"
    images are converting into pil format and save in specified name
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(out_dir, f"bedroom_{prefix}_{i:07d}.png"))

def read_images(lmdb_path):
    """
    working with an LMDB database to read and process images stored in WebP format
    """
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

TRAIN_LMBD_PATH = "/data2/LSUN_Bedroom/bedroom_train_lmdb"
VAL_LMBD_PATH = "/data2/LSUN_Bedroom/bedroom_val_lmdb"
TRAIN_OUT_DIR = "/data2/LSUN_Bedroom/train_orig_imgs"
VAL_OUT_DIR = "/data2/LSUN_Bedroom/val_orig_imgs"

t_images = read_images(TRAIN_LMBD_PATH)
dump_images(TRAIN_OUT_DIR, t_images, "train")
val_images = read_images(VAL_LMBD_PATH)
dump_images(VAL_OUT_DIR, val_images, "val")
