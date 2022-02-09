import sys
import math

from collections import Counter
from sacrebleu import corpus_bleu
from lexical_diversity import lex_div as ld
from fuzzywuzzy import fuzz

from mwzeval.utils import load_references
from mwzeval.database import MultiWOZVenueDatabase
from mwzeval.normalization import normalize_data

from mwzeval.utils import has_domain_predictions, get_domain_estimates_from_state
from mwzeval.utils import has_state_predictions
from mwzeval.utils import load_goals, load_booked_domains, load_gold_states


class Evaluator:

    def __init__(self, bleu : bool, success : bool, richness : bool, dst : bool = False):
        self.bleu = bleu
        self.success = success
        self.richness = richness
        self.dst = dst

        if bleu:
            self.reference_dialogs = load_references()

        if success:
            self.database = MultiWOZVenueDatabase()
            self.goals = load_goals()
            self.booked_domains = load_booked_domains()

        if dst:
            self.gold_states = load_gold_states() 

    def evaluate(self, input_data):
        normalize_data(input_data)
        return {
            "bleu"     : get_bleu(input_data, self.reference_dialogs)                             if self.bleu else None,
            "success"  : get_success(input_data, self.database, self.goals, self.booked_domains)  if self.success else None,
            "richness" : get_richness(input_data)                                                 if self.richness else None,
            "dst"      : get_dst(input_data, self.gold_states)                                    if self.dst else None,
        }


def get_bleu(input_data, reference_dialogs):
    """ Get SacreBLEU score between normalized utterances in input data and a set of normalized references. """

    hyps = []
    refs = {r : [] for r in reference_dialogs}

    for dialog_id, dialog in input_data.items():
        for turn_idx in range(len(dialog)):
            hyps.append(dialog[turn_idx]["response"])
            for r in refs:
                refs[r].append(reference_dialogs[r][dialog_id][turn_idx])

    return {r : corpus_bleu(hyps, [refs[r]]).score for r in refs}


def get_richness(input_data):
    """ Get lexical richness metrics (#unigrams, ..., entropies, turn lens., MSTTR) for normalized utterances in input data. """

    avg_lens, msttr, count = 0, 0, 0
    unique_grams = [Counter() for _ in range(3)]
    all_tokens = []

    for dialog in input_data.values():
        for turn in dialog:
            tokens = ld.tokenize(turn["response"])
            all_tokens.extend(tokens)
            
            avg_lens  += len(tokens)
            count += 1
            
            unique_grams[0].update(tokens)           
            unique_grams[1].update([(a, b) for a, b in zip(tokens, tokens[1:])])          
            unique_grams[2].update([(a, b, c) for a, b, c in zip(tokens, tokens[1:], tokens[2:])])
            
    avg_lens  /= count
    msttr = ld.msttr(all_tokens, window_length=50)      
    unique_grams_count = [len(c) for c in unique_grams]

    total = sum(v for v in unique_grams[0].values())
    probs = [(u/total) for u in unique_grams[0].values()]
    entropy = -sum(p * math.log(p, 2) for p in probs)
        
    cond = [unique_grams[1][(h, w)]/unique_grams[0][h] for h, w in unique_grams[1]]
    join = [unique_grams[1][(h, w)]/total for h, w in unique_grams[1]]
    cond_entropy = -sum(j * math.log(c, 2) for c, j in zip(cond, join))

    return {
        'entropy'         : entropy,
        'cond_entropy'    : cond_entropy,
        'avg_lengths'     : avg_lens,
        'msttr'           : msttr,
        'num_unigrams'    : unique_grams_count[0],
        'num_bigrams'     : unique_grams_count[1],
        'num_trigrams'    : unique_grams_count[2]
    }


def get_success(input_data, database, goals, booked_domains):
    """ Get Inform and Success rates of given dialogs, evaluate multiple setups: 
        1) Use predicted dialog state (if available, otherwise skip)
        2) Use ground-truth dialog state
        3) Use predicted dialog state (if available, otherwise skip) and optimistic scenario (use gold active domains, 
           match venues by intersection, ignore other search contraints if venue name or train id is present)
        4) Use ground-truth dialog state and optimistic scenario
    """
    
    if not has_state_predictions(input_data):
        #sys.stderr.write('warning: Missing state predictions, using ground-truth dialog states from MultiWOZ 2.2!\n')
        states = load_gold_states()  
        for dialog_id in input_data:
            for i, turn in enumerate(input_data[dialog_id]):
                turn["state"] = states[dialog_id][i]
    
    if not has_domain_predictions(input_data):
        #sys.stderr.write('warning: Missing domain predictions, estimating active domains from dialog states!\n')
        get_domain_estimates_from_state(input_data)    

    total = Counter()
    match_rate = {}
    success_rate = {}
    for dialog_id, dialog in input_data.items(): 
        
        utterances = [x["response"] for x in dialog]  
        dialog_states = [x["state"] for x in dialog]
        domain_estimates = [x["active_domains"] for x in dialog]  

        match, success = get_dialog_success(
            goals[dialog_id], 
            booked_domains[dialog_id], 
            utterances, 
            dialog_states, 
            domain_estimates, 
            database
        )

        for domain in set(match) | set(success):
            total[domain] += 1    
            match_rate[domain]   = match.get(domain, 0)   + match_rate.get(domain, 0)
            success_rate[domain] = success.get(domain, 0) + success_rate.get(domain, 0)

    match_rate   = {k : round(100 * match_rate[k]   / total[k], 1) for k in match_rate}
    success_rate = {k : round(100 * success_rate[k] / total[k], 1) for k in success_rate}

    return ({"inform" : match_rate, "success" : success_rate})


def get_dialog_success(goal, booked_domains, utterances, states, domain_estimates, database):
        
    requestable_slots = ['PHONE', 'ADDRESS', 'POST', 'REFERENCE', 'TRAINID']
    requestable_slots_in_goal = {d : set(goal[d]['requestable']) for d in goal}
    offered_venues = {d : [] for d in goal}
    provided_requestable_slots = {d : set() for d in goal}
    
    #
    # Find offered venues and provided requestable slots in system utterances
    
    for system_utterance, state, booked_domain, domain_estimate in zip(utterances, states, booked_domains, domain_estimates):
       
        for current_domain in goal:
        
            if current_domain not in domain_estimate:
                continue
            
            # in order to calculate the INFORM metric, we look at the NAME and TRAINID spans because these are the only
            # ones that identify a venue, search for the NAME or TRAINID in the current system response       
            if ('NAME' in system_utterance and current_domain in ['restaurant', 'hotel', 'attraction']) or ('TRAINID' in system_utterance and current_domain == 'train'):

                # The INFORM rate metric takes into account just the *last* mention about the particular venue into account
                matching_venues = database.query(current_domain, state[current_domain]) if current_domain in state else []

                # Go through the venues returned by the API call matching the dialog state of the current turn and
                # use them as the list of possibly offered venues if any of the possibly offered venues from the 
                # previous dialog turn is *not* present in the current list of matching venues, i.e., if it is not their subset
                if current_domain not in offered_venues or len(offered_venues[current_domain]) == 0:
                    offered_venues[current_domain] = matching_venues
                else:
                    if any(venue not in matching_venues for venue in offered_venues[current_domain]):
                        offered_venues[current_domain] = matching_venues

            for requestable_slot in requestable_slots:
                if requestable_slot in system_utterance:

                    # We do not want to add the REFERENCE to the set of mentioned slots if it could not have been known. But the MultiWOZ
                    # dataset does not provide any information about booking availability. Thus we need to rely on the ground-truth anotations. 
                    # On the other hand, if the system provides a reference code in the turn where it is not supposed to, the requestable slot 
                    # is not added. So even if the evaluated system does not use the ground-truth booking information during evaluation and 
                    # it always suceeds if it has any database results, it does not affect these metrics. 
                    if requestable_slot == 'REFERENCE':
                        if current_domain in booked_domain:
                            provided_requestable_slots[current_domain].add('REFERENCE')               
                    else: 
                        provided_requestable_slots[current_domain].add(requestable_slot)
                      
    for domain in goal:  
        
        # if the crowd worker was instructed to mention the name, the match is being done automatically
        if 'name' in goal[domain]['informable']:
            offered_venues[domain] = 'MATCHED'
        
        # 'taxi', 'police', 'hospital' are special domains - do not have any database and entity does not need to be provided
        if domain in ['taxi', 'police', 'hospital']:
            offered_venues[domain] = 'MATCHED'
        
        # if TRAINID was not requested and train was *not* found, we match the dialog goals, but if TRAINID was not requested
        # and train *was* found we want to keep it in order to check if it is a right train
        if domain == "train" and not offered_venues['train'] and 'TRAINID' not in goal['train']['requestable']:
            offered_venues[domain] = 'MATCHED'

    #        
    # Calculate the INFORM rate of this dialog, either +1 or 0  
    
    match = {}
    for domain in goal:
        match_domain = False
        
        if offered_venues[domain] == 'MATCHED':
            match_domain = True
        
        elif domain in ['restaurant', 'hotel', 'attraction', 'train'] and len(offered_venues[domain]) > 0:
        
            # Get venues from the database that match all the information provided by the user
            goal_venues = database.query(domain, goal[domain]['informable'])
                   
            # Compare the venues that could be offered by the system and the venues that match the information
            # in dialg goals, these two sets do not have to match exactly and there are two ways to compare them:
            # Venues are matching if the goal venues are a super set of the possibly offered venues.
            if set(offered_venues[domain]).issubset(set(goal_venues)):
                match_domain = True

        match[domain] = match_domain
 
    # The inform rate +1 if the goal venues are matched for all domains, otherwise 0 
    match['total'] = (sum(match.values()) == len(match.keys()))
    
    #
    # Calculate the SUCCESS rate, either +1 or 0
    
    success = {}
    if match['total']:
        for domain in goal:
         
            # if values in sentences are super set of requestables
            provided_and_wanted_slots = provided_requestable_slots[domain] & requestable_slots_in_goal[domain]
            domain_success = len(provided_and_wanted_slots) == len(requestable_slots_in_goal[domain])
            success[domain] = domain_success
            
        success['total'] = (sum(success.values()) >= len(success.keys()))

    return match, success


def get_dst(input_data, reference_states, fuzzy_ratio=95):
    """ Get dialog state tracking results: joint accuracy (exact state match), slot F1, precision and recall """
    
    def flatten(state_dict):
        constraints = {}
        for domain, state in state_dict.items():
            for s, v in state.items():
                constraints[(domain, s)] = v
        return constraints

    def is_matching(hyp, ref):
        hyp_k = hyp.keys()
        ref_k = ref.keys()
        if hyp_k != ref_k:
            return False
        for k in ref_k:
            if fuzz.partial_ratio(hyp[k], ref[k]) <= fuzzy_ratio:
                return False
        return True

    def compare(hyp, ref):
        # tp ... those mentioned in both and matching
        # tn ... those not mentioned in both (this inflates results for slot acc., thus reporting F1)
        # fn ... those not mentioned in hyp but mentioned in ref
        # fp ... those mentioned in hyp but not mentioned in ref OR mentioned in hyp but not matching
        tp, fp, fn = 0, 0, 0
        for slot, value in hyp.items():
            if slot in ref and fuzz.partial_ratio(value, ref[slot]) > fuzzy_ratio:
                tp += 1
            else:
                fp += 1
        for slot, value in ref.items():
            if slot not in hyp or fuzz.partial_ratio(hyp[slot], value) <= fuzzy_ratio:
                fn += 1
        return tp, fp, fn

    joint_match, slot_acc, slot_f1, slot_p, slot_r = 0, 0, 0, 0, 0

    if not has_state_predictions(input_data):
        sys.stderr.write('error: Missing state predictions!\n')

    else:
        total_tp, total_fp, total_fn = 0, 0, 0
        num_turns = 0
        for dialog_id in input_data:
            for i, turn in enumerate(input_data[dialog_id]):
                ref = flatten(reference_states[dialog_id][i])
                hyp = flatten(turn['state'])

                if is_matching(hyp, ref):
                    joint_match += 1
                
                tp, fp, fn = compare(hyp, ref)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                num_turns += 1

        slot_p = total_tp / (total_tp + total_fp + 1e-10)
        slot_r = total_tp / (total_tp + total_fn + 1e-10)
        slot_f1 = 2 * slot_p * slot_r / (slot_p + slot_r + 1e-10) * 100
        joint_match = joint_match / (num_turns + 1e-10) * 100

    return {
        'joint_accuracy'   : joint_match,
        'slot_f1'          : slot_f1,
        'slot_precision'   : slot_p,
        'slot_recall'      : slot_r
    }
