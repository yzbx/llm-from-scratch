import argparse
import concurrent.futures
import glob
import json
import os.path as osp
import time
from itertools import islice

import jsonlines
from deep_translator import GoogleTranslator
from tqdm import tqdm

from preprocess.start_redis import get_redis_database


def fun(work_id):
    r = get_redis_database()
    todo_size = r.llen('src_ids')
    print(f'todo size = {todo_size}')

    proxies_example = {
        "https": "192.168.28.130:2340",
        "http": "192.168.28.130:2340"
    }
    translator = GoogleTranslator(source='en', target='zh-CN',  proxies=proxies_example)

    if work_id == 0:
        tbar = tqdm(total = todo_size)

    while True:
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

                time.sleep(0.9)

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





