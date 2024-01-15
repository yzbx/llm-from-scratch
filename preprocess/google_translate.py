import argparse
import concurrent.futures
import glob
import json
import os.path as osp
import time
from itertools import islice

import jsonlines
from deep_translator import GoogleTranslator
from language_translate import VqaDataset
from tqdm import tqdm


def load_data():
    json_file = '/data/wangjiaxin/cvdataset/vqa/llava/llava_v1_5_mix665k.json'
    with open(json_file, 'r') as fp:
        data = json.load(fp)

    return data

def fun(work_id):
    data = load_data()
    vqa_dataset = VqaDataset(data[work_id::2])
    long_texts = vqa_dataset.long_texts
    todo_size = len(long_texts)
    print(f'todo size = {todo_size}')
    if todo_size == 0:
        return 0

    proxies_example = {
        "https": "192.168.28.130:2340",
        "http": "192.168.28.130:2340"
    }
    translator = GoogleTranslator(source='en', target='zh-CN',  proxies=proxies_example)

    if work_id == 0:
        tbar = tqdm(long_texts, total = todo_size)
    else:
        tbar = long_texts

    out_file = f'/data/wangjiaxin/cvdataset/vqa/llava/long_world2_rank{work_id}_llava_v1_5_mix665k.jsonl'
    writer = jsonlines.open(out_file, 'w', flush=False)

    for idx, value in enumerate(tbar):
        try:
            cn_value = translator.translate(value)
        except Exception as e:
            print(e)
            cn_value = ''

        writer.write(dict(en=value, cn=cn_value))
        if idx % 1024 == 0:
            writer._flush()

    writer.close()
    return 0

if __name__ == '__main__':
    max_workers = 2

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fun, range(max_workers)))

    print(f'results is {results}')





