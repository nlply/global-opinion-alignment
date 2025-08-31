import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import re

import ast

def alignment_score(D1, D2, Q):
    scores = []
    for q in Q:
        if q not in D1 or q not in D2:
            continue
        
        p1 = D1[q]
        p2 = D2[q]
        if len(p1) != len(p2):
            continue
        N = len(p1)  
        support = np.arange(1, N + 1) 
        wd = wasserstein_distance(support, support, p1, p2)
        
        score = 1 - wd / (N - 1)
        scores.append(score)
    return np.round(np.mean(scores), 4)

def our_score(lm_prob_dict, hm_dict, Q_list):
    total_score = 0.0
    count = 0
    for q_key in Q_list:
        if q_key not in lm_prob_dict or q_key not in hm_dict:
            continue
        count += 1
        lm_porb = lm_prob_dict[q_key]
        hm_prob = hm_dict[q_key]
        
        question_score = 1 - sum([np.abs(i-j) for i, j in zip(lm_porb, hm_prob)])/2
        total_score += question_score
    return total_score / count

def get_score_by_country(hm_df, country_dict):
    country_score_dict = dict()
    for code, name in country_dict.items():
        chm_df = hm_df[hm_df["B_COUNTRY"] == int(code)]
        if chm_df.empty:
            return

        hm_dict = dict()
        for Q_key, options in lm_option_dict.items():
            distribution_list = []
            for opt in options:
                distribution_list.append(len(chm_df[chm_df[Q_key] == int(opt)]))
            
            if sum(distribution_list) == 0:
                continue
            
            hm_dict[Q_key] = [d/sum(distribution_list) for d in distribution_list]
            
        for q in skip_dict[code]:
            if q in Q_list:
                Q_list.remove(q)

        country_score = alignment_score(lm_prob_dict, hm_dict, Q_list)
        if np.isnan(country_score.item()):
            pass
        else:   
            country_score_dict[name] = country_score
    return country_score_dict

skip_dict = {
    '32': ['Q40', 'Q60', 'Q80', 'Q150', 'Q160'],
    '152': ['Q40', 'Q60', 'Q80', 'Q150', 'Q160'],
    '858': ['Q40', 'Q60', 'Q80', 'Q150', 'Q160'],
    '156': ['Q60', 'Q70', 'Q110', 'Q150', 'Q160'],
    '392': ['Q60', 'Q70', 'Q90', 'Q110', 'Q130'],
    '410': ['Q40', 'Q80', 'Q150', 'Q160', 'Q170'],
    '276': ['Q40', 'Q80', 'Q150', 'Q160', 'Q170'],
    '643':  ['Q40', 'Q60', 'Q80', 'Q150', 'Q160'],
    '704': ['Q40', 'Q60', 'Q80', 'Q150', 'Q160'],
    '76': ['Q40', 'Q60', 'Q80', 'Q150', 'Q160'],
}

llm = "CohereLabs/aya-23-35B"
# llm = "meta-llama/Meta-Llama-3-70B-Instruct"
# llm = "Qwen/Qwen2.5-72B-Instruct"
# llm = "gpt-3.5-turbo"
# llm = "gpt-4"
# llm = "deepseek-chat"
# llm = "deepseek-reasoner"

country_list = ["Japan", "Korea", "Russia","Vietnam", "Brazil", "Argentina", "Chile", "Uruguay"]
country_code_dict = {
    "Argentina": 32,
    "Brazil": 76,
    "Chile": 152,
    "China": 156,
    "Germany": 276,
    "Japan": 392,
    "Korea": 410,
    "Russia": 643,
    "Uruguay": 858,
    "Vietnam": 704
}

for country in country_list:
    lm_data_path0 = f"results/RQ1/{llm}/results.csv" 

    print(f"********* {country} *********")

    lm_data_path1 = f"results/RQ2/language/{country}/{llm}/results.csv"
    lm_data_path2 = f"results/RQ2/persona/{country}/{llm}/results.csv" 
    lm_data_path3 = f"results/RQ2/language_persona/{country}/{llm}/results.csv" 

    lm_data_path4 = f"results/RQ2/few-shot/{country}/{llm}/results.csv" 
    lm_data_path5 = f"results/RQ2/language_few-shot/{country}/{llm}/results.csv" 

    lm_data_path_list = [lm_data_path0, lm_data_path1, lm_data_path2, lm_data_path3, lm_data_path4, lm_data_path5]
    
    for lm_data_path in lm_data_path_list:
        lm_df = pd.read_csv(lm_data_path, low_memory=False)
        lm_df = lm_df[lm_df['type'] == "Opinion-Dependent"]
        Q_list = []
        lm_prob_dict = dict()
        lm_option_dict = dict()
        skip_question_ids = []
        for index, row in lm_df.iterrows():
            Q_key = row['id']
            choice_keys = ast.literal_eval(row['choice_keys'])
            Q_list.append(Q_key)
            try:
                prob_dict = ast.literal_eval(row['response'])
            except:
                response = re.sub(r'(\d+):\s*([\d.]+%)', r"'\1': '\2'", row['response'])
                prob_dict = ast.literal_eval(response)

            d_clean = {k: float(v.strip('%')) / 100 for k, v in prob_dict.items()}

            for k in choice_keys:
                if k not in d_clean:
                    d_clean[k] = 0
            lm_prob_dict[Q_key] = [p for p in d_clean.values()]
            lm_option_dict[Q_key] = [k for k in d_clean.keys()]

        hm_data_path = "dataset/wvs/WV7.csv"
        hm_df = pd.read_csv(hm_data_path, low_memory=False)
        
        if country == "Argentina":
            country_dict = {
                "32": "Argentina",
            }
        elif country == "Brazil":
            country_dict = {
                "76": "Brazil",
            }
        elif country == "Chile":
            country_dict = {
                "152": "Chile",
            }
        elif country == "China":
            country_dict = {
                "156": "China",
            }
        elif country == "Germany":
            country_dict = {
                "276": "Germany",
            }
        elif country == "Japan":
            country_dict = {
                "392": "Japan",
            }
        elif country == "Korea":
            country_dict = {
                "410": "Korea",
            }
        elif country == "Russia":
            country_dict = {
                "643": "Russia",
            }
        elif country == "Uruguay":
            country_dict = {
                "858": "Uruguay",
            }
        elif country == "Vietnam":
            country_dict = {
                "704": "Vietnam",
            }
        
        country_score_dict = get_score_by_country(hm_df, country_dict)
        steering_method = ""
        if "/RQ1/" in lm_data_path:
            steering_method = "No Steering"
        elif "/language/" in lm_data_path:
            steering_method = "Language Steering"
        elif "/persona/" in lm_data_path:
            steering_method = "Persona Steering"
        elif "/language_persona/" in lm_data_path:
            steering_method = "Persona + Language Steering"
        elif "/few-shot/" in lm_data_path:
            steering_method = "Few-shot"
        elif "/language_few-shot/" in lm_data_path:
            steering_method = "Few-shot + Language Steering"

        for _, score in country_score_dict.items():
            print(f"{steering_method}: {score}")
            if "Language" in steering_method:
                print("-------------------")
        pass

