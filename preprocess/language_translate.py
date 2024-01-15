import argparse
import glob
import json
import os
import os.path as osp
from itertools import islice

import jsonlines
from opus_translator import OpusTranslator
from torch.utils.data import Dataset
from tqdm import tqdm

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def fun(translator, d):
    try:
        cn_values = translator.batch_translate([conv['value'] for conv in d['conversations']])
        for conv, cn_value in zip(d['conversations'], cn_values):
            conv['cn_value'] = cn_value

        d['translator'] = 'opus'
    except Exception as e:
        print(e)

    return d

class VqaDataset(Dataset):
    def __init__(self, data):
        super().__init__()

        texts = []
        long_texts = []
        for d in data:
            for x in d['conversations']:
                en_text = x['value']
                if len(en_text.split(' ')) >= 400:
                    long_texts.append(en_text)
                else:
                    texts.append(en_text)

        self.texts = texts
        self.long_texts = long_texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/data/wangjiaxin/cvdataset/vqa/llava')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    root_dir = args.root_dir

    translate_human_context = {}
    jfiles = glob.glob(f'{root_dir}/*.json')
    jfiles.sort()
    print(jfiles)
    for json_file in jfiles:
        with open(json_file, 'r') as fp:
            data = json.load(fp)

        out_file = osp.dirname(json_file) + f'/opus_world{args.world}_rank{args.rank}' + osp.basename(json_file) + 'l'
        writer = jsonlines.open(out_file, 'w', flush=True)

        translator = OpusTranslator(gpu=args.rank)

        vqa_dataset = VqaDataset(data[args.rank::args.world])

        for out in tqdm(translator.pipe(vqa_dataset, batch_size=args.batch_size)):
            writer.write_all(out)

        writer.close()





