import os

from argparse import ArgumentParser

from tqdm import tqdm
import re

import sys
sys.path.append(os.path.abspath("."))

from utils.wvs import WVS
from utils.common import (
    load_existing_ids, 
    save_result, 
    get_llm_expressed_distribution,
    get_prompt_of_rq2,
    match_llm_str,
    load_model,
    country_language_dict
)

os.environ["HF_HOME"] = "YOUR_CACHE_DIR"
os.environ["TRANSFORMERS_CACHE"] = "YOUR_CACHE_DIR"
download_dir = "YOUR_CACHE_DIR"

from huggingface_hub import login

token = "YOUR_TOKEN"
login(token=token)

def arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--country', type=str)
    parser.add_argument('--steering_method', choices=["language", "language_persona", "language_few-shot", "persona", "few-shot"], default="language")
    parser.add_argument('--cache_path', default='YOUR_CACHE_DIR')
    args = parser.parse_args()
    return args

def fix_prob_dict_str(s: str) -> str:
    return re.sub(r"(%)(\s*})$", r"\1'\2", s.strip())
    
def main(args):
    language = ""
    if "language" in args.steering_method:
        language = country_language_dict[args.country]
    else:
        language = "English"
    
    wvs = WVS("WV7", language)

    save_path = f"results/RQ2/{args.steering_method}/{args.country}/{args.model_name_or_path}/results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data_list = wvs.data_list
    
    existing_ids, item_list = load_existing_ids(
            save_path)
    
    for data in tqdm(data_list, total=len(data_list)):
        if data['id'] in existing_ids:
            continue
        existing_ids.add(data['id'])
        
        user_instruction = get_prompt_of_rq2(args, data)
        flag = True
        while flag:
            response = get_llm_expressed_distribution(args, user_instruction)
            response = match_llm_str(response)
            if response is None: flag = True
            else: flag = False

        item = [data['id'],
                data['question'],
                data['choices'],
                data['choice_keys'],
                data['answer'],
                data['type'],
                response]
        item_list.append(item)
        save_result(save_path, item)

if __name__ == '__main__':
    args = arguments()
    model,tokenizer = load_model(args)
    args.model = model
    args.tokenizer = tokenizer
    
    steering_method_list = ['language', 'language_persona', 'language_few-shot', 'few-shot', 'persona']
    country_list = [
        'China',
        'Japan',
        'Germany',
        'Korea',
        'Russia',
        'Vietnam',
        'Brazil',
        'Argentina',
        'Chile',
        'Uruguay'
    ]
    for country in country_list:
        args.country = country
        for steering_method in steering_method_list:
            args.steering_method = steering_method
            main(args)