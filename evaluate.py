# !/usr/bin/env python3
# conding=utf-8

import sys
import json

from mwzeval.metrics import Evaluator


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bleu", dest='bleu', action="store_true", default=False, help="If set, BLEU is evaluated.")
    parser.add_argument("-s", "--success", dest='success', action="store_true", default=False, help="If set, inform and success rates are evaluated.")
    parser.add_argument("-r", "--richness", dest='richness', action="store_true", default=False, help="If set, various lexical richness metrics are evaluated.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input JSON file path.")
    parser.add_argument("-o", "--output", type=str, default="evaluation_results.json", help="Output file path, here will be the final report.")
    args = parser.parse_args()

    if not args.bleu and not args.success and not args.richness:
        sys.stderr.write('error: Missing argument, at least one of -b, -s, and -r must be used!\n')
        parser.print_help()
        sys.exit(1)

    with open(args.input, 'r') as f:
        input_data = json.load(f)

    e = Evaluator(args.bleu, args.success, args.richness)
    results = e.evaluate(input_data)

    for metric, values in results.items():
        if values is not None:
            print(f"====== {metric.upper()} ======")
            for k, v in values.items():
                print(f"{k.ljust(15)}{v}")
            print("")

    with open(args.output, 'w+') as f:
        json.dump(results, f)
