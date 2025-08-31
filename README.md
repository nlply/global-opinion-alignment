> This repository contains the code and experimental results of the paper **On the Alignment of Large Language Models with Global Human Opinion**.

# Description of the folders
- **/code**: the code for the experiments
- **/utils**: the common utils for the code
- **/dataset**: the dataset used for the experiments
- **/inputs**: the few-shot examples for the experiments
- **/results**: the results of the experiments
- **/show_results**: the code for visualizing the results
- **/figures**: to save the figures

# No Steering
The [`prob_by_api.py`](code/RQ1/prob_by_api.py) and [`prob_by_llm.py`](code/RQ1/prob_by_llm.py) code files in the [code/RQ1](code/RQ1) folder can be used to run the "no steering" experiment. `prob_by_api.py` is used to run GPT-3.5, GPT-4, DeepSeek-V3, and DeepSeek-R1, while `prob_by_llm.py` is used to run Aya23, Llama3, and Qwen2.5.

Before running the code, please set `API_KEY` in [utils/common.py](utils/common.py).

The following script is an example of running code:
```
python code/RQ1/prob_by_api.py \
--model_name_or_path <model_name>
```
Here, `<model_name>` can be one of gpt-3.5-turbo, gpt-4, deepseek-chat, deepseek-reasoner.
```
python code/RQ1/prob_by_llm.py \
--model_name_or_path <model_name>
```
Here, `<model_name>` can be one of CohereLabs/aya-23-35B, meta-llama/Meta-Llama-3-70B-Instruct, Qwen/Qwen2.5-72B-Instruct.

After running the code, the results will be saved in results/RQ1/`<model_name>`/results.csv.

# Steering the LLMs
The [`prob_by_api.py`](code/RQ2/prob_by_api.py) and [`prob_by_llm.py`](code/RQ2/prob_by_llm.py) code files in the [code/RQ2](code/RQ2) folder can be used to run the "steering" experiment. Same as the "no steering" experiment, `prob_by_api.py` is used to run GPT-3.5, GPT-4, DeepSeek-V3, and DeepSeek-R1, while `prob_by_llm.py` is used to run Aya23, Llama3, and Qwen2.5.

Before running the code, you also need to set `API_KEY` in [utils/common.py](utils/common.py).

The following script is an example of running code:
```
python code/RQ2/prob_by_api.py \
--model_name_or_path <model_name> \ 
--country <country> \
--steering_method <steering_method>
```
Here, `<model_name>` can be one of gpt-3.5-turbo, gpt-4, deepseek-chat, deepseek-reasoner. `<country>` can be one of Argentina, Brazil, Chile, China, Germany, Japan,  Korea, Russia, Uruguay, Vietnam. `<steering_method>` can be one of persona, few-shot, language, language_persona, language_few-shot.

```
python code/RQ2/prob_by_llm.py \
--model_name_or_path <model_name> \ 
--country <country> \
--steering_method <steering_method>
```
Compare to `prob_by_api.py`, The only different is that `<model_name>` can be one of CohereLabs/aya-23-35B, meta-llama/Meta-Llama-3-70B-Instruct, Qwen/Qwen2.5-72B-Instruct.

After running the code, the results will be saved in results/RQ2/`<steering_method>`/`<country>`/`<model_name>`/results.csv.

**Note:** all the results already exist in the `results/RQ1/` and `results/RQ2/` folders, so you don't need to run the code again. If you are interested in the results, you can check the `results/RQ1/` and `results/RQ2/` folders and plot the results using the code under the [show_results](show_results) folder. 