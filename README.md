<h1 align="center">MultiWOZ Context-to-Response Evaluation</h1>

<h2 align="center">Standardized and easy to use Inform, Success, BLEU score</h2>

<p>&nbsp;</p>

_______

TODO: short description

# Usage

**Running:**

``` sh
python evaluate.py [--bleu] [--success] [--richness] --input INPUT.json [--output OUTPUT.json]
```
Add the `bleu` option to evaluate the BLEU score, `success` to get the Success and Inform rates, and use `richness` for getting lexical richness metrics such as the number of unique unigrams, trigrams, token entropy, bigram conditional entropy, corpus MSTTR-50, and average turn length. 


**Input format:**

``` json
{
    "xxx0000" : [
        {
            "response": "Your generated delexicalized response.",
            "state": {
                "restaurant" : {
                    "food" : "eatable"
                }, ...
            }, 
            "active_domains": ["restaurant"]
        }, ...
    ], ...
}
```
The input `.json` file should contain a dictionary with keys matching dialogue ids in the `xxx0000` format (e.g. `sng0073` instead of `SNG0073.json`), and values containing a list of turns. Each turn is a dictionary with keys:

- `response` – Your generated delexicalized response. You can use either the slot names with domain names, e.g. `restaurant_food`, or the domain adaptive delexicalization scheme, e.g. `food`.   
- `state` – **Optional**, the predicted dialog state. If not present (for example in case of policy optimization models), the ground truth dialog state from MultiWOZ 2.2 is used during the Inform and Success computation. Slot names and values are normalized prior the usage.
- `active_domains` – **Optional**, list of active domains for the corresponding turn. If not present, the active domains are estimated from changes in the dialog state during the Inform and Success rate computation. If your model predicts the domain for each turn, place them here. If you use domains in slot names, run the following command to extract the active domains from slot names automatically: 

    ``` sh
    python add_slot_domains.py [-h] -i INPUT.json -o OUTPUT.json
    ```

See the `predictions` folder for examples.

**Alternative usage directly from your code.** First instantiate an evaluator and then call the `evalute` method with dictionary of your predictions with the same format as describe above. Pseudo-code:

``` python
from mwz_evaluation.metrics import Evaluator
...

e = Evaluator(bleu=True, success=False, richness=False)
for epoch in data:
    ...
    results = e.evaluate(dict_of_my_predictions)
    print(f"Epoch {epoch} BLEU: {results}")
```

# Hall of fame

TODO: tables with results

# Documentation

TODO: description of methods

# Citation
`TBA`