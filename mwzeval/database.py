import os
import json

from fuzzywuzzy import fuzz
from mwzeval.normalization import normalize_state_slot_value, time_str_to_minutes


class MultiWOZVenueDatabase:
    
    IGNORE_VALUES = {
        'attraction' : ['location', 'openhours'],
        'hotel' : ['location', 'price', 'takesbookings'],
        'restaurant' : ['location', 'introduction', 'signature']
    }

    FUZZY_KEYS = {
        'hotel' : {'name'},
        'attraction' : {'name'},
        'restaurant' : {'name', 'food'},
        'train' : {'departure', 'destination'}
    }

    def __init__(self):
        self.data, self.data_keys = self._load_data()

    def _load_data(self):

        def normalize_column_name(name):
            if name == "id":
                return name
            name = name.lower().replace(' ', '')
            if name == "arriveby": return "arrive"
            if name == "leaveat": return "leave"
            return name

        database_data, database_keys = {}, {}
        
        for domain in ["restaurant", "attraction", "hotel", "train"]:
           
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, "data", "database", f"{domain}_db.json"), "r") as f:
                database_data[domain] = json.load(f)
            
            if domain in self.IGNORE_VALUES:
                for i in database_data[domain]:
                    for ignore in self.IGNORE_VALUES[domain]:
                        if ignore in i:
                            i.pop(ignore)

            for i, database_item in enumerate(database_data[domain]):
                database_data[domain][i] =  {normalize_column_name(k) : v for k, v in database_item.items()}
            
            database_keys[domain] = set(database_data[domain][0].keys())
            
        return database_data, database_keys

    def query(self, domain, constraints, fuzzy_ratio=90):

        # Hotel database keys:      address, area, name, phone, postcode, pricerange, type, internet, parking, stars (other are ignored)
        # Attraction database keys: address, area, name, phone, postcode, pricerange, type, entrance fee (other are ignored)
        # Restaurant database keys: address, area, name, phone, postcode, pricerange, type, food 
        # Train database contains keys: arriveby, departure, day, leaveat, destination, trainid, price, duration
        
        results = []
        
        if domain not in ["hotel", "restaurant", "attraction", "train"]:
            return results
        
        query = {}
        for key in self.data_keys[domain]:  
            if key in constraints:
                if constraints[key] in ["dontcare", "not mentioned", "don't care", "dont care", "do n't care", "do not care"]:
                    continue
                query[key] = normalize_state_slot_value(key, constraints[key])
                if key in ['arrive', 'leave']:
                    query[key] = time_str_to_minutes(query[key])
            else:
                query[key] = None
                        
        for i, item in enumerate(self.data[domain]):
            for k, v in query.items():
                if v is None or item[k] == '?':
                    continue

                if k == 'arrive':
                    time = time_str_to_minutes(item[k]) 
                    if time > v:
                        break
                elif k == 'leave':
                    time = time_str_to_minutes(item[k]) 
                    if time < v:
                        break
                else:
                    if k in self.FUZZY_KEYS.get(domain, {}):
                        f = (lambda x: fuzz.partial_ratio(item[k], x) < fuzzy_ratio)
                    else:
                        f = (lambda x: item[k] != x)
                    if f(v):
                        break
            else:
                if domain == "train":
                    results.append(item["trainid"])
                else:
                    results.append(item["id"])

        return results