"""
load dataset to redis
"""

import glob
import json

import jsonlines
import redis
from tqdm import tqdm


def get_redis_database():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True, charset='UTF-8', encoding='UTF-8')

    return r

def get_src_data():
    root_dir = '/data/wangjiaxin/cvdataset/vqa/llava'

    src_data = []
    for json_file in glob.glob(f'{root_dir}/*.json'):
        with open(json_file, 'r') as fp:
            data = json.load(fp)

        src_data += data
    return src_data

if __name__ == '__main__':
    r = get_redis_database()

    root_dir = '/data/wangjiaxin/cvdataset/vqa/llava'
    # basename = 'llava_v1_5_mix665k'
    basename = 'chat'
    src_data = []
    for json_file in glob.glob(f'{root_dir}/{basename}.json'):
        with open(json_file, 'r') as fp:
            data = json.load(fp)

        src_data += data

    print(f'found {len(src_data)} data')

    des_ids = set()
    for jfile in glob.glob(f'{root_dir}/zh_{basename}.jsonl'):
        with jsonlines.open(jfile, 'r') as reader:
            data = [obj for obj in reader]

        for obj in tqdm(data):
            des_ids.add(obj['id'])

    print(f'found {len(des_ids)} results')

    if r.exists('src_ids'):
        r.delete('src_ids')

    if r.exists('results'):
        r.delete('results')

    for obj in tqdm(src_data):
        if obj['id'] not in des_ids:
            r.rpush('src_ids', obj['id'])
            r.set(str(obj['id']), json.dumps(obj))

    print('save src_ids to redis')
