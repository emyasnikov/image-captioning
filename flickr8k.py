import concurrent.futures
import collections
import dataclasses
import hashlib
import itertools
import json
import math
import os
import pathlib
import random
import re
import string
import time
import urllib.request

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import requests
import tqdm

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds

def flickr8k(path="flickr8k"):
    path = pathlib.Path(path)

    if len(list(path.rglob("*"))) < 16197:
        tf.keras.utils.get_file(
            origin="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
            cache_dir=".",
            cache_subdir=path,
            extract=True,
        )
        tf.keras.utils.get_file(
            origin="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
            cache_dir=".",
            cache_subdir=path,
            extract=True,
        )

    captions = (path/"Flickr8k.token.txt").read_text().splitlines()
    captions = (line.split("\t") for line in captions)
    captions = ((filename.split("#")[0], caption) for (filename, caption) in captions)

    cap_dict = collections.defaultdict(list)

    for filename, caption in captions:
        cap_dict[filename].append(caption)

    train_files = (path/"Flickr_8k.trainImages.txt").read_text().splitlines()
    train_captions = [(str(path/"Flickr8k_Dataset"/filename), cap_dict[filename]) for filename in train_files]

    test_files = (path/"Flickr_8k.testImages.txt").read_text().splitlines()
    test_captions = [(str(path/"Flickr8k_Dataset"/filename), cap_dict[filename]) for filename in test_files]

    train_ds = tf.data.experimental.from_list(train_captions)
    test_ds = tf.data.experimental.from_list(test_captions)

    return train_ds, test_ds
