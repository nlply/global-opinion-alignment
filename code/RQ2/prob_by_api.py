import os

from argparse import ArgumentParser

from tqdm import tqdm

import sys
sys.path.append(os.path.abspath("."))

from utils.wvs import WVS
from utils.common import (
    load_existing_ids, 
    save_result, 
    get_api_expressed_distribution,
    get_prompt_of_rq2,
    match_api_str,
    country_language_dict
)

def arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--country', type=str)
    parser.add_argument('--steering_method', default="language_persona", 
                        choices=["persona", "few-shot", "language", "language_persona", "language_few-shot"])
    
    args = parser.parse_args()
    return args

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

    existing_ids, item_list = load_existing_ids(save_path)
    
    for data in tqdm(data_list, total=len(data_list)):
        if data['id'] in existing_ids:
            continue
        existing_ids.add(data['id'])

        user_instruction = get_prompt_of_rq2(args, data)
        flag = True
        while flag:
            response = get_api_expressed_distribution(args, user_instruction)
            response = match_api_str(response)
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
    model_list = [
        'gpt-3.5-turbo',
        'gpt-4',
        'deepseek-chat',
        'deepseek-reasoner',
    ]
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

    steer_method_list = [
        'persona',
        'few-shot',
        'language',
        'language_persona',
        'language_few-shot',
    ]
    for model in model_list:
        args.model_name_or_path = model
        for country in country_list:
            args.country = country
            for steer_method in steer_method_list:
                args.steering_method = steer_method
                main(args)