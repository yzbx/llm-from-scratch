import json
import time

import jsonlines
from tqdm import tqdm

from preprocess.start_redis import get_redis_database

if __name__ == '__main__':
    r = get_redis_database()
    root_dir = '/data/wangjiaxin/cvdataset/vqa/llava'
    basename = 'chat'

    out_jfile = f'{root_dir}/zh_{basename}.jsonl'
    writer = jsonlines.open(out_jfile, 'a', flush=True)

    while True:
        result = r.lpop('results')
        if result is None:
            time.sleep(0.1)
            continue

        d = json.loads(result)
        writer.write(d)

        to_save_size = r.llen('results')
        print(f'to_save_size = {to_save_size}')

    writer.close()
