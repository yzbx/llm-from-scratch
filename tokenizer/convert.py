from transformers import AutoTokenizer
import argparse
import json

def load_tokenizer(
        token_path_or_name: str
    ):
    """
    加载指定的tokenizer。
    """
    return AutoTokenizer.from_pretrained(
        token_path_or_name,
        trust_remote_code=True
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokenizer', help='tokenizer path or name')
    parser.add_argument('-o', '--output', help='output vocab json file', default='vocab.json')

    return parser.parse_args()

def main():
    args = get_args()
    tokenizer = load_tokenizer(args.tokenizer)

    with open(args.output, 'w') as fp:
        json.dump(tokenizer.get_vocab(), fp, ensure_ascii=False)
    return 0

if __name__ == '__main__':
    main()
