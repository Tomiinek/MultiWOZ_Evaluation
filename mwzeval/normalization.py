import re

from functools import partial
from sacremoses import MosesTokenizer, MosesDetokenizer


def normalize_data(input_data):
    """ In-place normalization of raw dictionary with input data. Normalize slot names, slot values, remove plurals and detokenize utterances. """

    mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
    slot_name_re = re.compile(r'\[([\w\s\d]+)\](es|s|-s|-es|)')
    slot_name_normalizer = partial(slot_name_re.sub, lambda x: normalize_slot_name(x.group(1)))

    for dialogue in input_data.values():
        for turn in dialogue:
            turn["response"] = slot_name_normalizer(turn["response"].lower())
            turn["response"] = md.detokenize(mt.tokenize(turn["response"].replace('-s', '').replace('-ly', '')))

            if "state" not in turn:
                continue

            for domain in turn["state"]:
                new_state = {}
                for slot, value in turn["state"][domain].items():          
                    slot = slot.lower().replace(' ', '')
                    if slot == "arriveby": slot = "arrive"
                    elif slot == "leaveat": slot = "leave"
                    new_state[slot] =  normalize_state_slot_value(slot, value)
                turn["state"][domain] = new_state
    

def normalize_slot_name(slot_name):
    """ Map a slot name to the new unified ontology. """

    slot_name = slot_name.lower()
    slot_name_mapping = {
     'ADDRESS'   : ['address', 'attraction_address', 'hospital_address', 'hotel_address', 'police_address', 'restaurant_address', 'value_address'],
     'AREA'      : ['area', 'value_area', 'attraction_area', 'restaurant_area', 'hotel_area'],
     'TIME'      : ['booktime', 'value_time', 'time', 'duration', 'value_duration', 'train_duration', 'arriveby', 'taxi_arriveby', 'value_arrive', 'arrive by', 'train_arriveby', 'leaveat', 'value_leave', 'leave at', 'train_leaveat', 'train_leave', 'train_arrive', 'taxi_leaveat'],
     'DAY'       : ['day', 'value_day', 'bookday', 'train_day'],
     'PLACE'     : ['destination', 'value_destination', 'departure', 'value_departure', 'value_place', 'train_departure', 'train_destination', 'taxi_destination', 'taxi_departure'],
     'FOOD'      : ['food', 'value_food', 'restaurant_food'],
     'NAME'      : ['name', 'attraction_name', 'hospital_name', 'hotel_name', 'police_name', 'restaurant_name', 'value_name'],
     'PHONE'     : ['phone', 'attraction_phone', 'hospital_phone', 'hotel_phone', 'police_phone', 'restaurant_phone', 'taxi_phone', 'value_phone'],
     'POST'      : ['postcode', 'attraction_postcode', 'hospital_postcode', 'hotel_postcode', 'restaurant_postcode', 'value_postcode', 'police_postcode'],
     'PRICE'     : ['price', 'value_price', 'entrancefee', 'entrance fee', 'train_price', 'attraction_entrancefee', 'pricerange', 'value_pricerange', 'price range', 'restaurant_pricerange', 'hotel_pricerange', 'attraction_pricerange', 'attraction_price'],
     'REFERENCE' : ['ref', 'attraction_reference', 'hotel_reference', 'restaurant_reference', 'train_reference', 'value_reference', 'reference'],  
     'COUNT'     : ['stars', 'value_stars', 'hotel_stars', 'bookstay', 'value_stay', 'stay', 'bookpeople', 'value_people', 'people', 'choice', 'value_choice', 'value_count', 'attraction_choice', 'hotel_choice', 'restaurant_choice', 'train_choice'],
     'TYPE'      : ['type', 'taxi_type', 'taxi_car', 'value_type', 'value_car', 'car', 'restaurant_type', 'hotel_type', 'attraction_type'],
     'TRAINID'   : ['trainid', 'train_id', 'value_id', 'id', 'train', 'train_trainid'],
     'INTERNET'  : ['internet', 'hotel_internet'],
     'PARKING'   : ['parking', 'hotel_parking'],
     'ID'        : ['hospital_id', 'attraction_id', 'restaurant_id'],
     'DEPARTMENT': ['value_department', 'department', 'hospital_department'],
     'OPEN'      : ['openhours']
    }
    reverse_slot_name_mapping = {s : k for k, v in slot_name_mapping.items() for s in v}  
    if slot_name not in reverse_slot_name_mapping:
        print(f"Unknown slot name: {slot_name}. Please use another slot names or customize the slot mapping!")
        return ''
    return reverse_slot_name_mapping[slot_name]


def normalize_state_slot_value(slot_name, value):
    """ Normalize slot value:
        1) replace too distant venue names with canonical values
        2) replace too distant food types with canonical values
        3) parse time strings to the HH:MM format
        4) resolve inconsistency between the database entries and parking and internet slots
    """
    
    def type_to_canonical(type_string): 
        if type_string == "swimming pool":
            return "swimmingpool" 
        elif type_string == "mutliple sports":
            return "multiple sports"
        elif type_string == "night club":
            return "nightclub"
        elif type_string == "guest house":
            return "guesthouse"
        return type_string

    def name_to_canonical(name, domain=None):
        """ Converts name to another form which is closer to the canonical form used in database. """

        name = name.strip().lower()
        name = name.replace(" & ", " and ")
        name = name.replace("&", " and ")
        name = name.replace(" '", "'")

        if domain is None or domain == "restaurant":
            if name == "hotel du vin bistro":
                return "hotel du vin and bistro"
            elif name == "the river bar and grill":
                return "the river bar steakhouse and grill"
            elif name == "nando's":
                return "nandos"
            elif name == "city center b and b":
                return "city center north b and b"
            elif name == "acorn house":
                return "acorn guest house"
            elif name == "caffee uno":
                return "caffe uno"
            elif name == "cafe uno":
                return "caffe uno"
            elif name == "rosa's":
                return "rosas bed and breakfast"
            elif name == "restaurant called two two":
                return "restaurant two two"
            elif name == "restaurant 2 two":
                return "restaurant two two"
            elif name == "restaurant two 2":
                return "restaurant two two"
            elif name == "restaurant 2 2":
                return "restaurant two two"
            elif name == "restaurant 1 7" or name == "restaurant 17":
                return "restaurant one seven"

        if domain is None or domain == "hotel":
            if name == "lime house":
                return "limehouse"
            elif name == "cityrooms":
                return "cityroomz"
            elif name == "whale of time":
                return "whale of a time"
            elif name == "huntingdon hotel":
                return "huntingdon marriott hotel"
            elif name == "holiday inn exlpress, cambridge":
                return "express by holiday inn cambridge"
            elif name == "university hotel":
                return "university arms hotel"
            elif name == "arbury guesthouse and lodge":
                return "arbury lodge guesthouse"
            elif name == "bridge house":
                return "bridge guest house"
            elif name == "arbury guesthouse":
                return "arbury lodge guesthouse"
            elif name == "nandos in the city centre":
                return "nandos city centre"

        if domain is None or domain == "attraction":
            if name == "broughton gallery":
                return "broughton house gallery"
            elif name == "scudamores punt co":
                return "scudamores punting co"
            elif name == "cambridge botanic gardens":
                return "cambridge university botanic gardens"
            elif name == "the junction":
                return "junction theatre"
            elif name == "trinity street college":
                return "trinity college"
            elif name in ['christ college', 'christs']:
                return "christ's college"
            elif name == "history of science museum":
                return "whipple museum of the history of science"
            elif name == "parkside pools":
                return "parkside swimming pool"
            elif name == "the botanical gardens at cambridge university":
                return "cambridge university botanic gardens"
            elif name == "cafe jello museum":
                return "cafe jello gallery"

        return name

    def time_to_canonical(time):
        """ Converts time to the only format supported by database, e.g. 07:15. """
        time = time.strip().lower()

        if time == "afternoon": return "13:00"
        if time == "lunch" or time == "noon" or time == "mid-day" or time == "around lunch time": return "12:00"
        if time == "morning": return "08:00"
        if time.startswith("one o'clock p.m"): return "13:00"
        if time.startswith("ten o'clock a.m"): return "10:00"
        if time == "seven o'clock tomorrow evening":  return "07:00"
        if time == "three forty five p.m":  return "15:45"
        if time == "one thirty p.m.":  return "13:30"
        if time == "six fourty five":  return "06:45"
        if time == "eight thirty":  return "08:30"

        if time.startswith("by"):
            time = time[3:]

        if time.startswith("after"):
            time = time[5:].strip()

        if time.startswith("afer"):
            time = time[4:].strip()    

        if time.endswith("am"):   time = time[:-2].strip()
        if time.endswith("a.m."): time = time[:-4].strip()

        if time.endswith("pm") or time.endswith("p.m."):
            if time.endswith("pm"):   time = time[:-2].strip()
            if time.endswith("p.m."): time = time[:-4].strip()
            tokens = time.split(':')
            if len(tokens) == 2:
                return str(int(tokens[0]) + 12) + ':' + tokens[1] 
            if len(tokens) == 1 and tokens[0].isdigit():
                return str(int(tokens[0]) + 12) + ':00'
        
        if len(time) == 0:
            return "00:00"
            
        if time[-1] == '.' or time[-1] == ',' or time[-1] == '?':
            time = time[:-1]
            
        if time.isdigit() and len(time) == 4:
            return time[:2] + ':' + time[2:]

        if time.isdigit(): return time.zfill(2) + ":00"
        
        if ':' in time:
            time = ''.join(time.split(' '))

        if len(time) == 4 and time[1] == ':':
            tokens = time.split(':')
            return tokens[0].zfill(2) + ':' + tokens[1]

        return time

    def food_to_canonical(food):
        """ Converts food name to caninical form used in database. """

        food = food.strip().lower()

        if food == "eriterean": return "mediterranean"
        if food == "brazilian": return "portuguese"
        if food == "sea food": return "seafood"
        if food == "portugese": return "portuguese"
        if food == "modern american": return "north american"
        if food == "americas": return "north american"
        if food == "intalian": return "italian"
        if food == "italain": return "italian"
        if food == "asian or oriental": return "asian"
        if food == "english": return "british"
        if food == "australasian": return "australian"
        if food == "gastropod": return "gastropub"
        if food == "brutish": return "british"
        if food == "bristish": return "british"
        if food == "europeon": return "european"

        return food

    if slot_name in ["name", "destination", "departure"]:
        return name_to_canonical(value)
    elif slot_name == "type":
        return type_to_canonical(value)
    elif slot_name == "food":
        return food_to_canonical(value)
    elif slot_name in ["arrive", "leave", "arriveby", "leaveat", "time"]:
        return time_to_canonical(value)
    elif slot_name in ["parking", "internet"]:
        return "yes" if value == "free" else value
    else:
        return value


def time_str_to_minutes(time_string):
    if not re.match(r"[0-9][0-9]:[0-9][0-9]", time_string):
        return 0
    return int(time_string.split(':')[0]) * 60 + int(time_string.split(':')[1])
