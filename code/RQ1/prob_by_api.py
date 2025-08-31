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
    get_prompt_of_rq1,
    match_api_str
)

def arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--temperature', type=float, default=0)
    args = parser.parse_args()
    return args
    
def main(args):
    wvs = WVS("WV7", "English")
    save_path = f"results/RQ1/{args.model_name_or_path}/results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data_list = wvs.data_list
    existing_ids, item_list = load_existing_ids(save_path)
    
    for data in tqdm(data_list, total=len(data_list)):
        if data['id'] in existing_ids:
            continue
        existing_ids.add(data['id'])

        user_instruction = get_prompt_of_rq1(data)
        
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

    for model in model_list:
        args.model_name_or_path = model
        main(args)