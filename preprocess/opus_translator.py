import argparse
import concurrent.futures
import glob
import json
import os.path as osp
import time
from itertools import islice
from typing import List

import jsonlines
from tqdm import tqdm
from transformers import pipeline

from preprocess.start_redis import get_redis_database


class OpusTranslator():
    def __init__(self, gpu=0):
        self.pipe = pipeline("translation", model="/data/wangjiaxin/huggingface/opus-mt-en-zh", trust_remote_code=True, device=f'cuda:{gpu}')

    def translate(self, text: str) -> str:
        result = self.pipe(text)
        return result[0]['translation_text']

    def batch_translate(self, texts: List[str]) -> List[str]:
        results = self.pipe(texts)

        return [result['translation_text'] for result in results]

def fun(work_id):
    r = get_redis_database()
    todo_size = r.llen('src_ids')
    print(f'todo size = {todo_size}')

    translator = OpusTranslator(gpu=work_id)

    idx = 0
    if work_id == 0:
        tbar = tqdm(total = todo_size)

    while True:
        remain_size = r.llen('src_ids')
        if remain_size == 0:
            print('nothing in src_ids list')
            break 

        src_id = r.lpop('src_ids')

        if src_id is None:
            print('src_id is None')
            break

        r_data = r.get(str(src_id))
        if r_data is None:
            print(f'r_data is None for {src_id}')
            break

        d = json.loads(r_data)
        try:
            for conv in d['conversations']:
                value = conv['value']
                cn_value = translator.translate(value)
                conv['cn_value'] = cn_value

                time.sleep(1)

            d['translator'] = 'opus'
            r.rpush('results', json.dumps(d))
        except Exception as e:
            print(e)

        if work_id == 0:
            remain_size = r.llen('src_ids')
            finished = todo_size - remain_size
            tbar.update(finished)

    return 0

if __name__ == '__main__':
    max_workers = 2

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fun, range(max_workers)))

    print(f'results is {results}')



