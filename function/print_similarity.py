import re
import pandas as pd
from tabulate import tabulate
from typing import Dict, Any

def find_word(text, word):
    result = re.findall('\\b'+word+'\\b', text, flags=re.IGNORECASE)
    return len(result) > 0


def print_similarity(keywords, dists, classes) -> pd.DataFrame:
    result = {
        "Keyword": keywords,
        "Score": dists.cpu().numpy(),
        "Acc." : [],
        "Bias" : []
    }

    for keyword, diff in zip(keywords, dists):
        biased_index =  classes['caption'].apply(find_word, word=keyword) 
        biased_dataset = classes[biased_index]  
        biased = biased_dataset.shape[0]
        correct_of_biased = sum(biased_dataset['actual'] == biased_dataset['pred'])
        # correct_of_biased = sum(biased_dataset['correct'])
        biased_accuracy = correct_of_biased / biased
        result["Acc."].append(biased_accuracy)
        if diff < 0:
            result["Bias"].append("")
        else:
            result["Bias"].append("S")

    diff = pd.DataFrame(result)
    diff = diff.sort_values(by = ["Score"], ascending = False)
    print(tabulate(diff, headers='keys', showindex=False))
    return diff