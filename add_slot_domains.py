# !/usr/bin/env python3
# conding=utf-8

import sys
import re
import json

from functools import partial
from mwzeval.utils import has_domain_predictions


def extract_active_domains_from_slot_names(data):
    
    slot_name_re = re.compile(r'\[([\w\s\d]+)\]')
    get_slots = partial(slot_name_re.sub, lambda x: domains.add(x.group(1).split('_')[0]))

    for dialog_id, dialog in data.items():
        for turn in dialog:
            domains = set()
            get_slots(turn['response'].lower())
            turn["active_domains"] = [x for x in domains if x in ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input JSON file path.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file path.")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        input_data = json.load(f)

    if has_domain_predictions(input_data):
        print('warning: Domain information in: {args.input} will be overwritten!')
        while True:
            usr = input("Do you want to continue? [Y/n]")
            if usr in ["y", "Y", ""]:
                break
            elif usr in ["n", "N"]:
                sys.exit(0)

    extract_active_domains_from_slot_names(input_data)

    with open(args.output, 'w+') as f:
        json.dump(input_data, f, indent=2)