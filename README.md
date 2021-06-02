<h1 align="center">MultiWOZ Context-to-Response Evaluation</h1>

<h3 align="center">Standardized and easy to use Inform, Success, BLEU</h3>

<p>&nbsp;</p>

_______

Easy-to-use scripts for standardized evaluation of response generation on the [MultiWOZ benchmark](https://github.com/budzianowski/multiwoz). This repository contains an implementation of the MultiWOZ database with fuzzy matching, functions for normalization of slot names and values, and a careful implementation of the BLEU score and Inform & Succes rates. 

# Usage

**Getting started:**

``` sh
git clone https://github.com/Tomiinek/MultiWOZ_Evaluation.git && cd MultiWOZ_Evaluation
pip install -r requirements.txt
```

**Running:**

``` sh
python evaluate.py [--bleu] [--success] [--richness] --input INPUT.json [--output OUTPUT.json]
```
Add `--bleu` to evaluate the BLEU score, `--success` to get the Success & Inform rate, and use `--richness` for getting lexical richness metrics such as the number of unique unigrams, trigrams, token entropy, bigram conditional entropy, corpus MSTTR-50, and average turn length. 


**Input format:**

``` python
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

- **`response`** – Your generated delexicalized response. You can use either the slot names with domain names, e.g. `restaurant_food`, or the domain adaptive delexicalization scheme, e.g. `food`.   
- **`state`** – **Optional**, the predicted dialog state. If not present (for example in case of policy optimization models), the ground truth dialog state from MultiWOZ 2.2 is used during the Inform & Success computation. Slot names and values are normalized prior the usage.
- **`active_domains`** – **Optional**, list of active domains for the corresponding turn. If not present, the active domains are estimated from changes in the dialog state during the Inform & Success rate computation. If your model predicts the domain for each turn, place them here. If you use domains in slot names, run the following command to extract the active domains from slot names automatically: 

    ``` sh
    python add_slot_domains.py [-h] -i INPUT.json -o OUTPUT.json
    ```

See the [`predictions`](predictions) folder with examples.

**Alternative usage directly from your code.** First instantiate an evaluator and then call the `evalute` method with dictionary of your predictions with the same format as describe above. Pseudo-code:

``` python
from multiwoz_evaluation.metrics import Evaluator
...

e = Evaluator(bleu=True, success=False, richness=False)
my_predictions = {}
for item in data:
    my_predictions[item.dialog_id] = model.predict(item)
    ...
    
results = e.evaluate(my_predictions)
print(f"Epoch {epoch} BLEU: {results}")
```


**Output format:**

``` python
{
    "bleu" : {'damd': … , 'uniconv': … , 'hdsa': … , 'lava': … , 'augpt': … , 'mwz22': … },
    "success" : {
        "inform"  : {'attraction': … , 'hotel': … , 'restaurant': … , 'taxi': … , 'total': … , 'train': … },
        "success" : {'attraction': … , 'hotel': … , 'restaurant': … , 'taxi': … , 'total': … , 'train': … },
    },
    "richness" : {
        'entropy': … , 'cond_entropy': … , 'avg_lengths': … , 'msttr': … , 
        'num_unigrams': … , 'num_bigrams': … , 'num_trigrams': … 
    }
}
```
The evaluation script outputs a dictionary with keys `bleu`, `success`, and `richness` corresponding to BLEU, Inform & Success rates, and lexical richness metrics, respectively. Their values can be `None` if not evaluated, otherwise: 

- **BLEU** results contain multiple scores corresponding to different delexicalization styles and refernces. Currently included references are DAMD, HDSA, AuGPT, LAVA, UniConv, and **MultiWOZ 2.2** whitch we consider to be the canonical one that should be reported in the future. 
- **Inform & Succes** rates are reported for each domain (i.e. attraction, restaurant, hotel, taxi, and train in case of the test set) separately and in total.
- **Lexical richness** contains the number of distinct uni-, bi-, and tri-grams, average number of tokens in generated responses, token entropy, conditional bigram entropy, and MSTTR-50 calculated on concatenated responses.


# Hall of Fame

- See the [`predictions`](predictions) folder for details about the raw generated predictions.
- BLEU reported in these tables is calculated with references obtained from the *MultiWOZ 2.2 span annotations*.
- CBE stands for *conditional bigram entropy*. 

-------------------

| Model              | BLEU | Inform  | Success  | Avg. len. | CBE | #uniq. words | #uniq. 3-grams |
| ------------------ | -----:| -------:| --------:| ---------:| -----------------:| -------------:| -------------:| 
| Reference corpus &nbsp; | -    | 93.7 | 90.9 | 14.00 | 3.01 | 1407 | 23877 | 

**End-to-end models**, i.e. those that use only the context as input.

| Model              | BLEU | Inform  | Success  | Avg. len. | CBE | #uniq. words | #uniq. 3-grams |
| ------------------ | -----:| -------:| --------:| ---------:| -----------------:| -------------:| -------------:| 
| DAMD ([paper](https://arxiv.org/abs/1911.10484)\|[code](https://github.com/thu-spmi/damd-multiwoz))  | 16.4 | 57.9 | 47.6 | 14.27 | 1.65 | 212  | 1755  |
| MinTL ([paper](https://arxiv.org/pdf/2009.12005.pdf)\|[code](https://github.com/zlinao/MinTL)) | **19.4** | 73.7 | 65.4 | 14.78 | 1.81 | 297  | 2525  |
| UBAR ([paper](https://arxiv.org/abs/2012.03539)\|[code](https://github.com/TonyNemo/UBAR-MultiWOZ))  | 17.6 | **83.4** | 70.3 | 13.54 | 2.10 | 478  | 5238  |
| SOLOIST ([paper](https://arxiv.org/abs/2005.05298))  | 13.6 | 82.3 | **72.4** | 18.45 | **2.41** | **615**  | **7923**  |
| AuGPT ([paper](https://arxiv.org/abs/2102.05126)\|[code](https://github.com/ufal/augpt)) | 16.8 | 76.6 | 60.5 | 12.90 | 2.15 | 608  | 5843  |
| LABES ([paper](https://arxiv.org/pdf/2009.08115v3.pdf)\|[code](https://github.com/thu-spmi/LABES)) | 18.9 | 68.5 | 58.1 | 14.20 | 1.83 | 374  | 3228  |
| DoTS ([paper](https://arxiv.org/pdf/2103.06648.pdf))  | 16.8 | 80.4 | 68.7 | 14.66 | 2.10 | 411  | 5162  |

**Policy optimization models**, i.e. those that use also the ground-truth dialog states.

| Model              | BLEU | Inform  | Success  | Avg. len. | CBE | #uniq. words | #uniq. 3-grams |
| ------------------ | -----:| -------:| --------:| ---------:| -----------------:| -------------:| -------------:|
| MarCo ([paper](https://arxiv.org/pdf/2004.12363.pdf)\|[code](https://github.com/InitialBug/MarCo-Dialog))   | 17.3 | 94.5 | 87.2 | 16.01 | **1.94** | 319 | **3002** | 
| HDSA ([paper](https://arxiv.org/pdf/1905.12866.pdf)\|[code](https://github.com/wenhuchen/HDSA-Dialog))    | **20.7** | 87.9 | 79.4 | 14.42 | 1.64 | 259 | 2019 |
| HDNO ([paper](https://arxiv.org/pdf/2006.06814.pdf)\|[code](https://github.com/mikezhang95/HDNO))    | 17.8 | 93.3 | 83.4 | 14.96 | 0.84 | 103 | 315  |
| SFN ([paper](https://arxiv.org/pdf/1907.10016.pdf)\|[code](https://github.com/Shikib/structured_fusion_networks))     | 14.1 | 93.4 | 82.3 | 14.93 | 1.63 | 188 | 1218 |
| UniConv ([paper](https://arxiv.org/pdf/2004.14307.pdf)\|[code](https://github.com/henryhungle/UniConv)) | 18.1 | 66.7 | 58.7 | 14.17 | 1.79 | **338** | 2932 |
| LAVA ([paper](https://arxiv.org/abs/2011.09378)\|[code](https://gitlab.cs.uni-duesseldorf.de/general/dsml/lava-public/-/tree/master/experiments_woz/sys_config_log_model/2020-05-12-14-51-49-actz_cat))    | 10.8 | **95.9** | **93.5** | 13.28 | 1.27 | 176 | 708  |

# Contributing

- **If you would like to add your results into the Hall of Fame**, modify the particular table, add the file with predictions into the `predictions` folder, and create a pull request.
- **If you need to update the [slot name mapping](https://github.com/Tomiinek/MultiWOZ_Evaluation/blob/29512dec6df009e6b579a4aa8d26f8c1c6e85e35/normalization.py#L36-L55)** because of your different delexicalization style, feel free to make the changes, and create a pull request.
- **If you would like to improve [normalization of slot values](https://github.com/Tomiinek/MultiWOZ_Evaluation/blob/29512dec6df009e6b579a4aa8d26f8c1c6e85e35/normalization.py#L63-L254)**, add your new rules, and create a pull request.

# Citation
`TBA`
