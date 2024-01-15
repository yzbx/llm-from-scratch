import argparse
import concurrent.futures
import glob
import json
import os.path as osp
import time
from itertools import islice

import jsonlines
from deep_translator import BaiduTranslator
from tqdm import tqdm

from preprocess.start_redis import get_redis_database


def fun(work_id):
    r = get_redis_database()
    todo_size = r.llen('src_ids')
    print(f'todo size = {todo_size}')

    baidu_keys = {
        '20240111001937990': 'ClnRDYI0WZh2ym_Wswze',
        # '20240111001938148': 'crSdvB7GC6wZMFvUGYh5',
        # '20230309001593485': '9DTeiRtlMXD1eq63uMdh',
        # '20240111001938163': 'u6mNp8hMkNAHaBXjHry_',
        # '20240111001938161': '7YdIrxUoQAZ4HhsaQFV3',
        # '20240111001938158': 'SQybqQwfyiRXzLA0Rxdp',
        # '20240111001938189': 'So1ktAbIap2317HesboH',
        # '20240111001938200': 'xOHEEkMQ2lDhYRwlZ60M',
    }
    id_keys = [item for item in baidu_keys.items()]
    app_id, app_key = id_keys[work_id]

    translator = BaiduTranslator(appid=app_id, appkey=app_key, source="en", target="zh")

    idx = 0
    print(f'{work_id} with {app_id} and {app_key}')
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

                time.sleep(1)

            r.rpush('results', json.dumps(d))
        except Exception as e:
            print(e)

        if work_id == 0 and idx % 128 == 0:
            todo_size = r.llen('src_ids')
            print(f'{work_id} {idx}/{todo_size}')

        idx += 1
        if work_id == 0:
            tbar.update(idx)

    return 0

if __name__ == '__main__':
    max_workers = 2

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fun, range(max_workers)))

    print(f'results is {results}')



