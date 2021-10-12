import os
import json
import urllib.request

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


def load_references(systems=['mwz22']): #, 'damd', 'uniconv', 'hdsa', 'lava', 'augpt']):
    references = {}
    for system in systems:
        if system == 'mwz22':
            continue
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "data", "references", f"{system}.json")) as f:
            references[system] = json.load(f)
    if 'mwz22' in systems:
        references['mwz22'] = load_multiwoz22_reference()
    return references


def load_multiwoz22_reference():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "data", "references", "mwz22.json")
    if os.path.exists(data_path):
        with open(data_path) as f:
            return json.load(f)
    references, _ = load_multiwoz22()
    return references


def load_gold_states():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "data", "gold_states.json")
    if os.path.exists(data_path):
        with open(data_path) as f:
            return json.load(f)
    _, states = load_multiwoz22()
    return states

    
def load_multiwoz22():

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

    normalize_data(mwz22_data)
    
    references, states = {}, {}
    for dialog in mwz22_data:
        references[dialog] = [x["response"] for x  in mwz22_data[dialog]]
        states[dialog] = [x["state"] for x  in mwz22_data[dialog]]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    reference_path = os.path.join(dir_path, "data", "references", "mwz22.json")
    state_path = os.path.join(dir_path, "data", "gold_states.json")

    with open(reference_path, 'w+') as f:
        json.dump(references, f, indent=2)

    with open(state_path, 'w+') as f:
        json.dump(states, f, indent=2)

    return references, states
