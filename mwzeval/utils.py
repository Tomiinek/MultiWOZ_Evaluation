import os
import json
import urllib.request
from typing import Literal
import zipfile
import io
from collections import defaultdict

from mwzeval.normalization import normalize_data


def has_domain_predictions(data):
    for dialog in data.values():
        for turn in dialog:
            if "active_domains" not in turn:
                return False
    return True


def get_domain_estimates_from_state(data):

    for dialog in data.values():

        # Use an approximation of the current domain because the slot names used for delexicalization do not contain any
        # information about the domain they belong to. However, it is likely that the system talks about the same domain
        # as the domain that recently changed in the dialog state (which should be probably used for the possible lexicalization). 
        # Moreover, the usage of the domain removes a very strong assumption done in the original evaluation script assuming that 
        # all requestable slots are mentioned only and exactly for one domain (through the whole dialog).

        current_domain = None
        old_state = {}
        old_changed_domains = []

        for turn in dialog:
 
            # Find all domains that changed, i.e. their set of slot name, slot value pairs changed.
            changed_domains = []
            for domain in turn["state"]:
                domain_state_difference = set(turn["state"].get(domain, {}).items()) - set(old_state.get(domain, {}).items())
                if len(domain_state_difference) > 0:
                    changed_domains.append(domain)

            # Update the current domain with the domain whose state currently changed, if multiple domains were changed then:
            # - if the old current domain also changed, let the current domain be
            # - if the old current domain did not change, overwrite it with the changed domain with most filled slots
            # - if there were multiple domains in the last turn and we kept the old current domain & there are currently no changed domains, use the other old domain
            if len(changed_domains) == 0:
                if current_domain is None:
                    turn["active_domains"] = []
                    continue 
                else:
                    if len(old_changed_domains) > 1:
                        old_changed_domains = [x for x in old_changed_domains if x in turn["state"] and x != current_domain]
                        if len(old_changed_domains) > 0:
                            current_domain = old_changed_domains[0] 

            elif current_domain not in changed_domains:
                current_domain = max(changed_domains, key=lambda x: len(turn["state"][x]))

            old_state = turn["state"]
            old_changed_domains = changed_domains
            
            turn["active_domains"] = [current_domain]


def has_state_predictions(data):
    for dialog in data.values():
        for turn in dialog:
            if "state" not in turn:
                return False
    return True


def load_goals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "data", "goals.json")) as f:
        return json.load(f)


def load_booked_domains():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "data", "booked_domains.json")) as f:
        return json.load(f)


def load_references(systems=['mwz22'], enable_normalization: bool = True): #, 'damd', 'uniconv', 'hdsa', 'lava', 'augpt']):
    references = {}
    for system in systems:
        if system == 'mwz22':
            continue
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "data", "references", f"{system}.json")) as f:
            references[system] = json.load(f)
    if 'mwz22' in systems:
        references['mwz22'] = load_multiwoz22_reference(enable_normalization=enable_normalization)
    return references


def load_multiwoz22_reference(enable_normalization: bool = True):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "data", "references", "mwz22.json" if enable_normalization else "mwz22_not_normalized.json")
    if os.path.exists(data_path):
        with open(data_path) as f:
            return json.load(f)
    references, _ = load_multiwoz22(enable_normalization=enable_normalization)
    return references


def load_gold_states(mwz_version: Literal['22', '24'] = '22', enable_normalization: bool = True):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "data", f"gold_states{mwz_version}.json" if enable_normalization else f"gold_states{mwz_version}_not_normalized.json")
    if os.path.exists(data_path):
        with open(data_path) as f:
            return json.load(f)
    if mwz_version == "22":
        _, states = load_multiwoz22(enable_normalization=enable_normalization)
    elif mwz_version == "24":
        _, states = load_multiwoz24(enable_normalization=enable_normalization)
    else:
        raise ValueError("Unsupported MultiWOZ version.")
    return states

    
def load_multiwoz22(enable_normalization: bool = True):

    def delexicalize_utterance(utterance, span_info):
        span_info.sort(key=(lambda  x: x[-2])) # sort spans by start index
        new_utterance = ""
        prev_start = 0
        for span in span_info:
            intent, slot_name, value, start, end = span
            if start < prev_start or value == "dontcare":
                continue
            new_utterance += utterance[prev_start:start]
            new_utterance += f"[{slot_name}]"
            prev_start = end
        new_utterance += utterance[prev_start:]
        return new_utterance

    def parse_state(turn):
        state = {}
        for frame in turn["frames"]:  
            domain = frame["service"]
            domain_state = {}
            slots = frame["state"]["slot_values"]
            for name, value in slots.items():
                if "dontcare" in value:
                    continue 
                domain_state[name.split('-')[1]] = value[0]
            
            if domain_state:
                state[domain] = domain_state
            
        return state

    with urllib.request.urlopen("https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/dialog_acts.json") as url:
        print("Downloading MultiWOZ_2.2/dialog_act.json ")
        dialog_acts = json.loads(url.read().decode())

    raw_data = []
    folds = {
        "train" : 17, 
        "dev"   : 2, 
        "test"  : 2
    }
    for f, n in folds.items():
        for i in range(n):
            print(f"Downloading MultiWOZ_2.2/{f}/dialogues_{str(i+1).zfill(3)}.json ")
            with urllib.request.urlopen(f"https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/{f}/dialogues_{str(i+1).zfill(3)}.json") as url:
                raw_data.extend(json.loads(url.read().decode()))

    mwz22_data = {}
    for dialog in raw_data:
        parsed_turns = []
        for i in range(len(dialog["turns"])):
            t = dialog["turns"][i]
            if i % 2 == 0:
                state = parse_state(t)
                continue       
            parsed_turns.append({
                "response" : delexicalize_utterance(t["utterance"], dialog_acts[dialog["dialogue_id"]][t["turn_id"]]["span_info"]),
                "state" : state
            })           
        mwz22_data[dialog["dialogue_id"].split('.')[0].lower()] = parsed_turns

    if enable_normalization:
        normalize_data(mwz22_data)
    
    references, states = {}, {}
    for dialog in mwz22_data:
        references[dialog] = [x["response"] for x  in mwz22_data[dialog]]
        states[dialog] = [x["state"] for x  in mwz22_data[dialog]]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    reference_path = os.path.join(dir_path, "data", "references", "mwz22.json" if enable_normalization else "mwz22_not_normalized.json")
    state_path = os.path.join(dir_path, "data", "gold_states22.json" if enable_normalization else "gold_states22_not_normalized.json")

    with open(reference_path, 'w+') as f:
        json.dump(references, f, indent=2)

    with open(state_path, 'w+') as f:
        json.dump(states, f, indent=2)

    return references, states


def load_multiwoz24(enable_normalization: bool = True):
    def is_filled(slot_value: str, consider_none_as_filled: bool = False) -> bool:
        """Whether a slot value is filled.

        Unfilled slots should be dropped, as in MultiWOZ 2.2.

        Args:
            slot_value: the value to check
            consider_none_as_filled: whether slots with value "none" should be considered as filled
        """
        slot_value = slot_value.lower()
        return slot_value and slot_value != "not mentioned" and (slot_value != "none" or consider_none_as_filled)

    def get_first_value(values: str) -> str:
        """Get the first value if the values string contains multiple."""
        if "|" in values:
            values = values.split("|")
        elif ">" in values:
            values = values.split(">")
        elif "<" in values:
            values = values.split("<")
        else:
            values = [values]
        return values[0]

    def parse_state(turn: dict, prepend_book: bool = False) -> dict[dict[str, str]]:
        """Get the slot values of a given turn.

        This function is adapted from
        google-research/schema_guided_dst/multiwoz/create_data_from_multiwoz.py

        If a slot has multiple values (which are separated by '|', '<' or '>'), only the first one is taken.
        This is consistant with the approach taken for MultiWOZ 2.2 evaluation.

        Args:
            turn: Dictionary of a turn of the MultiWOZ 2.4 dataset
            prepend_book: Whether to prepend the string 'book' to slot names for booking slots.
                MultiWOZ 2.2 has the 'book' prefix.

        Returns:
            {$domain: {$slot_name: $value, ...}, ...}
        """
        dialog_states = defaultdict(dict)
        for domain_name, values in turn['metadata'].items():
            domain_dial_state = {}

            for k, v in values["book"].items():
                # Note: "booked" is not really a state, just booking confirmation
                if k == 'booked':
                    continue
                if isinstance(v, list):
                    for item_dict in v:
                        new_states = {
                            (f"book{slot_name}" if prepend_book else slot_name): slot_val
                            for slot_name, slot_val in item_dict.items()
                        }
                        domain_dial_state.update(new_states)
                if isinstance(v, str) and v:
                    slot_name = f"book{k}" if prepend_book else k
                    domain_dial_state[slot_name] = v

            new_states = values["semi"]
            domain_dial_state.update(new_states)

            domain_dial_state = {
                slot_name: get_first_value(value)  # use the first value
                for slot_name, value in domain_dial_state.items()
                if is_filled(value)
            }
            if len(domain_dial_state) > 0:
                dialog_states[domain_name] = domain_dial_state

        return dialog_states

    with urllib.request.urlopen(
        "https://github.com/smartyfh/MultiWOZ2.4/blob/main/data/MULTIWOZ2.4.zip?raw=true"
    ) as url:
        print("Downloading MultiWOZ_2.4")
        unzipped = zipfile.ZipFile(io.BytesIO(url.read()))
        # dialogue_acts = json.loads(unzipped.read('MULTIWOZ2.4/dialogue_acts.json'))
        data = json.loads(unzipped.read('MULTIWOZ2.4/data.json'))

    mwz24_data = {}
    for dialogue_id, dialogue in data.items():
        parsed_turns = []
        for i, turn in enumerate(dialogue["log"]):
            if i % 2 == 0:
                continue
            state = parse_state(turn)
            parsed_turns.append({"response": "", "state": state})
        mwz24_data[dialogue_id.split(".")[0].lower()] = parsed_turns

    if enable_normalization:
        normalize_data(mwz24_data)

    references, states = {}, {}
    for dialog in mwz24_data:
        #  references[dialog] = [x["response"] for x in mwz24_data[dialog]]
        states[dialog] = [x["state"] for x in mwz24_data[dialog]]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    #  reference_path = os.path.join(dir_path, "data", "references", "mwz24.json" if enable_normalization else "mwz24_not_normalized.json")
    state_path = os.path.join(dir_path, "data", "gold_states24.json" if enable_normalization else "gold_states24_not_normalized.json")

    #  with open(reference_path, 'w+') as f:
        #  json.dump(references, f, indent=2)

    with open(state_path, 'w+') as f:
        json.dump(states, f, indent=2)

    return references, states
