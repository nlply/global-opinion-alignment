import os
import csv
import ast
import re

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_existing_ids(file_path):
    """
        load existing ids
    """
    existing_ids = set()
    item_list = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_ids.add(row['id'])
                item_list.append([row['id'],
                                  row['question'],
                                  row['choices'],
                                  row['choice_keys'],
                                  row['answer'],
                                  row['type'],
                                  row['response']])
    else:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['id',
                      'question',
                      'choices',
                      'choice_keys',
                      'answer',
                      'type',
                      'response']
            writer.writerow(header)
    return existing_ids, item_list
    

def save_result(save_result_path, item):
    """
        save result
    """
    with open(save_result_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(item)

def get_api_expressed_distribution(args, user_instruction):
    """
        get expressed distribution
    """
    if args.model_name_or_path.startswith("deepseek-"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_instruction}]
        client = OpenAI(api_key="YOUR_API_KEY",
                    base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=messages,
                temperature=args.temperature,
                stream=False
            )
        model_response=response

    else:
        api_key = "YOUR_API_KEY"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_instruction}]
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=messages,
                logprobs=None,
                top_logprobs=None,
                temperature=args.temperature)
        model_response=response
        print(response)
    try: response = model_response.choices[0].message.content.split('Answer: ')[1]
    except: response = model_response.choices[0].message.content
    return response

def get_llm_expressed_distribution(args, user_instruction):
    model = args.model
    tokenizer = args.tokenizer
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_instruction}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
        
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, 
                                 return_dict_in_generate=True,
                                 output_scores=True,
                                 max_new_tokens=256,
                                 do_sample=False)
    generated_token_ids = outputs.sequences[0][inputs['input_ids'].shape[-1]:]
    predicted_token = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    response = predicted_token.strip()
    return response

def read_file_to_string(file_path):
    """
        read few shot examples from file_path
    """
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print("File not found.")
        return None
    
def get_prompt_of_rq1(data):
    """
        get prompt of rq1
    """
    prompt = "I will provide a distribution over answer choices on a series of questions to help you understand the formatting of this task. After the 5 examples, please express the distribution of answers for the question asked. Do not follow the distribution expressed in the examples as they are only to give you an example of formatting, express the distribution based on your understanding of how humans may respond. Please only respond in the exact format of a dictionary mapping answer choice number to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. "
    prompt+='First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself.'
    prompt+= read_file_to_string('inputs/few_shot/lang-En_dist-random.txt')
    prompt+='\nYour turn! '
    prompt += "\nQuestion: " + data['question']
    prompt += "\n" + data['answer']
    prompt+="\nAnswer:"
    return prompt

def match_api_str(text):
    """
        match the output
    """
    pattern = r"\{(?:\s*'\d+'\s*:\s*'\d+(?:\.\d+)?%'\s*,?)+\s*\}"
    match = re.search(pattern, text)
    if match:
        probs_str = match.group(0)
        prob_dict = ast.literal_eval(probs_str)

        probs = {k: float(v.strip('%')) / 100 for k, v in prob_dict.items()}
        if abs(sum(probs.values()) - 1.0) < 5e-1:
            return probs_str
        else:
            return None
        
    elif re.search(pattern, re.sub(r'(\d+):\s*([\d.]+%)', r"'\1': '\2'", text)):
        match = re.search(pattern, re.sub(r'(\d+):\s*([\d.]+%)', r"'\1': '\2'", text))
        probs_str = match.group(0)
        prob_dict = ast.literal_eval(probs_str)

        probs = {k: float(v.strip('%')) / 100 for k, v in prob_dict.items()}
        if abs(sum(probs.values()) - 1.0) < 5e-1:
            return probs_str
        else:
            return None
    elif ast.literal_eval(text):
        prob_dict = ast.literal_eval(text)

        probs = {k: float(v.strip('%')) / 100 for k, v in prob_dict.items()}
        if abs(sum(probs.values()) - 1.0) < 5e-1:
            return text
        else:
            return None
    else:
        return None

def normalize(s):
    raw = {k: float(v.strip('%')) for k, v in ast.literal_eval(s).items()}
    total = sum(raw.values())
    probs_pct = {k: f"{v/total*100:.2f}%" for k, v in raw.items()}
    return str(probs_pct)

def match_llm_str(text):
    pattern = r"\{(?:\s*'?\d+'?\s*:\s*'\d+(?:\.\d+)?%'\s*,?)+\s*\}"
    if '%}' in text:
        text = text.replace('%}', '%\'}')
    text = re.sub(r"(?<!%)([0-9])'}", r"\1%'}", text)
    text = re.sub(r"(?<!%)([0-9])',", r"\1%',", text)
    match = re.search(pattern, text)
    if match:
        probs_str = match.group(0)
        prob_dict = ast.literal_eval(probs_str)

        probs = {k: float(v.strip('%')) / 100 for k, v in prob_dict.items()}
        if abs(sum(probs.values()) - 1.0) < 5e-1:
            return probs_str
        else:     
            normalized_probs_str = normalize(probs_str)
            return normalized_probs_str
    else:
        print("Not matched:", text)
        return None
     
def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 torch_dtype=torch.float16, 
                                                    device_map="auto", 
                                                    offload_folder="offload",
                                                    trust_remote_code=True,
                                                    cache_dir=args.cache_path)
    return model, tokenizer


country_language_dict = {
    "China": "Chinese",
    "Japan": "Japanese",
    "Germany": "German",
    "Korea": "Korean",
    "Russia": "Russian",
    "Vietnam": "Vietnamese",
    "Brazil": "Portuguese",
    "Argentina": "Spanish",
    "Chile": "Spanish",
    "Uruguay": "Spanish",
}
country_human_dict = {
    "China": "Chinese",
    "Japan": "Japanese",
    "Germany": "German",
    "Korea": "Korean",
    "Russia": "Russian",
    "Vietnam": "Vietnamese",
    "Brazil": "Brazilian",
    "Argentina": "Argentinian",
    "Chile": "Chilean",
    "Uruguay": "Uruguayan",
}

def get_prompt_of_language(country, data):
    prompt = ""
    if country == "China":
        prompt += "我将提供一系列问题的答案选项分布，以帮助您理解本任务的格式。在5个示例之后，请根据您对人类可能如何回答的理解，表达该问题的答案分布。请不要遵循示例中表达的分布，因为它们仅用于示例格式，请根据您对人类可能如何回答的理解表达分布。请仅以字典映射格式回答，即将答案选项编号与概率映射，不包含额外空格、字符、引号或文本。请仅以这种格式生成1句话。不符合此格式的答案将不被接受。"
        prompt += "首先，我将提供几个该任务的示例，以便您了解答案的格式要求，随后您将自行完成该任务。"
        prompt += read_file_to_string('inputs/few_shot/lang-Zh_dist-random.txt')
        prompt += "\n您的回答： "
        prompt += "\n问题: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\n答案:"
    elif country == "Japan":
        prompt += "このタスクのフォーマットを理解いただくため、一連の質問に対する回答選択肢の分布を示します。5つの例示の後、質問された内容に対する回答の分布を表現してください。例示で示された分布に従わないでください。これらはフォーマットの例を示すためのものであり、人間の回答パターンを基に、ご自身の理解に基づいて分布を表現してください。辞書形式で、回答選択肢の番号と確率を対応させる正確な形式で回答してください。余分なスペース、文字、引用符、またはテキストは含まないでください。この形式で1文のみを提出してください。この形式外の回答は受け付けられません。"
        prompt += "まず、このタスクの例をいくつか示し、回答のフォーマットのイメージを掴んでいただきます。その後、ご自身でタスクを実施していただきます。"
        prompt += read_file_to_string('inputs/few_shot/lang-Ja_dist-random.txt')
        prompt += "\nあなたの番です！ "
        prompt += "\n質問: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\n回答:"
    elif country == "Germany":
        prompt += "Ich werde Ihnen eine Verteilung der Antwortmöglichkeiten für eine Reihe von Fragen zur Verfügung stellen, damit Sie das Format dieser Aufgabe besser verstehen können. Nach den 5 Beispielen geben Sie bitte die Verteilung der Antworten für die gestellte Frage an. Folgen Sie nicht der in den Beispielen angegebenen Verteilung, da diese nur dazu dient, Ihnen ein Beispiel für das Format zu geben. Geben Sie die Verteilung basierend auf Ihrem Verständnis davon an, wie Menschen antworten könnten. Bitte antworten Sie nur in dem genauen Format eines Wörterbuchs, in dem die Antwortauswahlnummer der Wahrscheinlichkeit zugeordnet ist, ohne zusätzliche Leerzeichen, Zeichen, Anführungszeichen oder Text. Bitte verfassen Sie nur einen Satz in diesem Format. Antworten außerhalb dieses Formats werden nicht akzeptiert. "
        prompt += "Zunächst werde ich Ihnen einige Beispiele für diese Aufgabe geben, damit Sie ein Gefühl für die Formatierung der Antwort bekommen. Anschließend werden Sie die Aufgabe selbst ausführen."
        prompt += read_file_to_string('inputs/few_shot/lang-De_dist-random.txt')
        prompt += "\nDu bist dran! "
        prompt += "\nFrage: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nAntwort:"
    elif country == "Korea":
        prompt += "이 작업의 형식을 이해하는 데 도움을 드리기 위해 일련의 질문에 대한 답변 선택지의 분포를 제공하겠습니다. 5개의 예시 이후에는 제시된 질문에 대한 답변의 분포를 표현해 주시기 바랍니다. 예시에서 표현된 분포를 따르지 마십시오. 예시는 단순히 형식을 보여주기 위한 것이며, 인간이 어떻게 응답할지 이해한 대로 분포를 표현해 주시기 바랍니다. 답변은 반드시 사전 형식으로, 답변 선택지 번호와 확률을 매핑하는 형식으로만 작성해 주시기 바랍니다. 추가 공백, 문자, 따옴표 또는 텍스트는 포함하지 마십시오. 이 형식으로 단 한 문장만 작성해 주시기 바랍니다. 이 형식 외의 답변은 수용되지 않습니다. "
        prompt += "먼저 이 작업의 몇 가지 예를 들어 답변 형식을 이해하시도록 도와드리겠습니다. 그 다음에는 직접 작업을 수행해 보시게 될 것입니다."
        prompt += read_file_to_string('inputs/few_shot/lang-Ko_dist-random.txt')
        prompt +="\n이제 당신의 차례입니다! "
        prompt += "\n문제: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\n답변:"
    elif country == "Russia":
        prompt += "Я предоставлю распределения по вариантам ответов для ряда вопросов, чтобы помочь вам понять формат этой задачи. После 5 примеров, опираясь на ваше понимание того, как могли бы отвечать люди, выразите распределение по вариантам для данного вопроса. Пожалуйста, не следуйте распределениям из примеров — они предназначены только для демонстрации формата; укажите распределение согласно вашему пониманию того, как отвечали бы люди. Отвечайте только в формате отображения словаря, то есть сопоставьте номера вариантов ответов с вероятностями, без дополнительных пробелов, символов, кавычек или текста. Сгенерируйте ровно одну строку в этом формате. Ответы, не соответствующие этому формату, не будут приняты."
        prompt += "Сначала я приведу несколько примеров этой задачи, чтобы вы поняли требования к формату ответа, после чего вы выполните задачу самостоятельно."
        prompt += read_file_to_string('inputs/few_shot/lang-Ru_dist-random.txt')
        prompt += "\nВаш ответ: "
        prompt += "\nВопрос: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nОтвет:"
    elif country == "Vietnam":
        prompt += "Tôi sẽ cung cấp phân bố các phương án trả lời cho một loạt câu hỏi để giúp bạn hiểu định dạng của nhiệm vụ này. Sau 5 ví dụ, hãy biểu diễn phân bố câu trả lời cho câu hỏi dựa trên hiểu biết của bạn về cách con người có thể trả lời. Vui lòng không làm theo các phân bố trong ví dụ, vì chúng chỉ dùng để minh họa định dạng; hãy biểu diễn phân bố theo hiểu biết của bạn về cách con người có thể trả lời. Chỉ được trả lời theo định dạng ánh xạ từ điển, tức là ánh xạ số thứ tự phương án với xác suất, không chứa khoảng trắng thừa, ký tự, dấu ngoặc kép hoặc văn bản bổ sung. Chỉ tạo đúng 1 dòng theo định dạng này. Câu trả lời không đúng định dạng sẽ bị từ chối."
        prompt += "Trước hết, tôi sẽ cung cấp một vài ví dụ của nhiệm vụ này để bạn hiểu yêu cầu về định dạng câu trả lời, sau đó bạn sẽ tự hoàn thành nhiệm vụ."
        prompt += read_file_to_string('inputs/few_shot/lang-Vi_dist-random.txt')
        prompt += "\nCâu trả lời của bạn: "
        prompt += "\nCâu hỏi: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nĐáp án:"
    elif country == "Brazil":
        prompt += "Vou fornecer a distribuição das opções de resposta de uma série de perguntas, para ajudá-lo a entender o formato desta tarefa. Após 5 exemplos, expresse a distribuição das respostas para a questão dada com base no seu entendimento de como humanos provavelmente responderiam. Não siga as distribuições apresentadas nos exemplos, pois elas servem apenas para ilustrar o formato; expresse a distribuição de acordo com o seu entendimento de como humanos provavelmente responderiam. Responda apenas no formato de mapeamento de dicionário, isto é, mapeando o número de cada opção de resposta para a sua probabilidade, sem espaços extras, caracteres, aspas ou texto adicionais. Gere apenas 1 frase nesse formato. Respostas que não estiverem nesse formato não serão aceitas."
        prompt += "Primeiro, fornecerei alguns exemplos desta tarefa para que você entenda os requisitos de formatação das respostas; em seguida, você concluirá a tarefa por conta própria."
        prompt += read_file_to_string('inputs/few_shot/lang-Pt_dist-random.txt')
        prompt += "\nSua resposta: "
        prompt += "\nPergunta: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nResposta:"
    elif country in ["Argentina", "Chile", "Uruguay"]:
        prompt += "Voy a proporcionar la distribución de opciones de respuesta para una serie de preguntas, para ayudarle a comprender el formato de esta tarea. Después de cinco ejemplos, por favor exprese la distribución de respuestas para la pregunta según su comprensión de cómo podrían responder los humanos. No siga las distribuciones mostradas en los ejemplos, ya que son solo para ilustrar el formato; exprese la distribución basándose en su comprensión de cómo podrían responder los humanos. Por favor, responda únicamente en el formato de mapeo de diccionario, asignando el número de cada opción de respuesta a su probabilidad, sin espacios, caracteres, comillas o texto adicionales. Genere solo una frase en este formato. Las respuestas que no cumplan este formato no serán aceptadas."
        prompt += "Primero, proporcionaré algunos ejemplos de esta tarea para que comprenda los requisitos de formato de las respuestas; luego completará la tarea por su cuenta."
        prompt += read_file_to_string('inputs/few_shot/lang-Es_dist-random.txt')
        prompt += "\nSu respuesta: "
        prompt += "\nPregunta: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nRespuesta:"
    return prompt

def get_prompt_of_language_persona(country, data):
    prompt = ""
    if country == "China":
        prompt += "我将提供一系列问题的答案选项分布，以帮助您理解本任务的格式。在5个示例之后，请根据您对人类可能如何回答的理解，表达一组中国人对所提问题的答案分布。请不要遵循示例中表达的分布，因为它们仅用于示例格式，请根据您对人类可能如何回答的理解表达分布。请仅以字典映射格式回答，即将答案选项编号与概率映射，不包含额外空格、字符、引号或文本。请仅以这种格式生成1句话。不符合此格式的答案将不被接受。 "
        prompt += "首先，我将提供几个该任务的示例，以便您了解答案的格式要求，随后您将自行完成该任务，针对中国人。"
        prompt += read_file_to_string('inputs/few_shot/lang-Zh_dist-random.txt')
        prompt += "\n轮到你了！ "
        prompt += "\n问题: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\n答案:"
    elif country == "Japan":
        prompt += "このタスクのフォーマットを理解いただくため、一連の質問に対する回答選択肢の分布を示します。5つの例示の後、質問された内容に対する日本人のグループからの回答の分布を表現してください。例示で示された分布に従わないでください。これらはフォーマットの例を示すためのものであり、人間の回答パターンを基に、ご自身の理解に基づいて分布を表現してください。辞書形式で回答選択肢の番号と確率を対応付ける正確な形式で回答してください。余分なスペース、文字、引用符、またはテキストは含まないでください。この形式で1文のみを提出してください。この形式外の回答は受け付けられません。 ".format(args.culture)
        prompt += "まず、このタスクの例をいくつか示し、回答のフォーマットのイメージを掴んでいただきます。その後、ご自身で日本人を対象にタスクを実施していただきます。"
        prompt += read_file_to_string('inputs/few_shot/lang-Ja_dist-random.txt')
        prompt += "\nあなたの番です！ "
        prompt += "\n質問: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\n回答:"
    elif country == "Germany":
        prompt += "Ich werde Ihnen eine Reihe von Antwortoptionen für Fragen zur Verfügung stellen, um Ihnen zu helfen, das Format dieser Aufgabe zu verstehen. Nach den fünf Beispielen bitten wir Sie, die Antwortverteilung einer Gruppe von Deutschen auf die gestellten Fragen entsprechend Ihrer Einschätzung, wie Menschen antworten könnten, darzustellen. Bitte folgen Sie nicht den in den Beispielen dargestellten Verteilungen, da diese nur als Beispiel für das Format dienen. Geben Sie die Verteilung entsprechend Ihrer Vorstellung davon an, wie Menschen antworten könnten. Bitte antworten Sie ausschließlich im Dictionary-Mapping-Format, d. h. ordnen Sie die Antwortoptionen den Wahrscheinlichkeiten zu, ohne zusätzliche Leerzeichen, Zeichen, Anführungszeichen oder Text. Bitte erstellen Sie nur einen Satz in diesem Format. Antworten, die diesem Format nicht entsprechen, werden nicht akzeptiert. "
        prompt += "Zunächst werde ich Ihnen einige Beispiele für diese Aufgabe geben, damit Sie die Anforderungen an das Format der Antworten verstehen. Anschließend werden Sie die Aufgabe selbstständig für Personen aus Deutschland bearbeiten."
        prompt += read_file_to_string('inputs/few_shot/lang-De_dist-random.txt')
        prompt += "\nDu bist dran! "
        prompt += "\nFrage: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nAntwort:"
    elif country == "Korea":
        prompt += "저는 이 작업의 형식을 이해하는 데 도움을 드리기 위해 일련의 질문에 대한 답변 옵션 분포를 제공하겠습니다. 5개의 예시 이후에는, 인간이 어떻게 답변할지 이해하신 대로 한국인들이 해당 질문에 대한 답변 분포를 표현해 주시기 바랍니다. 예시에서 표현된 분포를 따르지 마십시오. 예시는 형식만을 보여주기 위한 것이며, 인간이 어떻게 답변할지 이해한 대로 분포를 표현해 주시기 바랍니다. 답변은 사전 매핑 형식으로만 작성해 주시기 바랍니다. 즉, 답변 옵션 번호와 확률을 매핑하며, 추가 공백, 문자, 따옴표 또는 텍스트를 포함하지 않아야 합니다. 이 형식으로 단 한 문장만 생성해 주시기 바랍니다. 이 형식을 준수하지 않은 답변은 수용되지 않습니다. "
        prompt += "먼저, 해당 작업의 예시를 몇 가지 제공하여 답변의 형식 요건을 이해하시도록 하겠습니다. 이후에는 해당 작업을 직접 수행해 주시되, 대상은 한국인입니다."
        prompt += read_file_to_string('inputs/few_shot/lang-Ko_dist-random.txt')
        prompt += "\n이제 당신의 차례입니다! "
        prompt += "\n문제: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\n답변:"
    elif country == "Russia":
        prompt += "Я предоставлю распределения по вариантам ответов для ряда вопросов, чтобы помочь вам понять формат этой задачи. После 5 примеров, исходя из вашего понимания того, как люди могут отвечать, выразите распределение ответов группы россиян на заданные вопросы. Пожалуйста, не следуйте распределениям в примерах, поскольку они служат только для демонстрации формата; укажите распределение на основе вашего понимания того, как люди могут отвечать. Отвечайте только в виде отображения (словаря), сопоставляющего номера вариантов ответов с вероятностями, без лишних пробелов, символов, кавычек или текста. Сгенерируйте ровно одну строку в этом формате. Ответы, не соответствующие этому формату, не будут приняты. "
        prompt += "Сначала я приведу несколько примеров по этой задаче, чтобы вы поняли требования к формату ответа, после чего вы самостоятельно выполните задачу для россиян."
        prompt += read_file_to_string('inputs/few_shot/lang-Ru_dist-random.txt')
        prompt += "\nТеперь ваша очередь! "
        prompt += "\nВопрос: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nОтвет:"
    elif country == "Vietnam":
        prompt += "Tôi sẽ cung cấp phân bố các lựa chọn trả lời cho một loạt câu hỏi để giúp bạn hiểu định dạng của nhiệm vụ này. Sau 5 ví dụ, dựa trên hiểu biết của bạn về cách con người có thể trả lời, hãy biểu đạt phân bố câu trả lời của một nhóm người Việt Nam cho câu hỏi đã nêu. Xin đừng làm theo các phân bố trong ví dụ, vì chúng chỉ dùng để minh họa định dạng; hãy đưa ra phân bố dựa trên hiểu biết của bạn về cách con người có thể trả lời. Chỉ trả lời theo định dạng ánh xạ từ điển, tức ánh xạ số thứ tự lựa chọn với xác suất, không chứa khoảng trắng, ký tự, dấu ngoặc kép hoặc văn bản thừa. Chỉ tạo 1 câu theo định dạng này. Câu trả lời không đúng định dạng sẽ không được chấp nhận. "
        prompt += "Trước hết, tôi sẽ cung cấp một vài ví dụ của nhiệm vụ này để bạn nắm yêu cầu về định dạng câu trả lời; sau đó bạn sẽ tự hoàn thành nhiệm vụ, đối với người Việt Nam."
        prompt += read_file_to_string('inputs/few_shot/lang-Vi_dist-random.txt')
        prompt += "\nĐến lượt bạn! "
        prompt += "\nCâu hỏi: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nĐáp án:"
    elif country == "Brazil":
        prompt += "Vou fornecer a distribuição das opções de resposta para uma série de perguntas, para ajudar você a entender o formato desta tarefa. Após 5 exemplos, com base no seu entendimento de como os seres humanos provavelmente responderiam, expresse a distribuição das respostas para um conjunto de brasileiros às perguntas apresentadas. Não siga as distribuições mostradas nos exemplos, pois elas servem apenas para ilustrar o formato; expresse as distribuições de acordo com o seu entendimento de como as pessoas provavelmente responderiam. Responda somente no formato de mapeamento de dicionário, isto é, mapeando o número de cada opção de resposta para a sua probabilidade, sem espaços extras, caracteres, aspas ou texto adicional. Gere apenas 1 frase nesse formato. Respostas fora desse formato não serão aceitas. "
        prompt += "Primeiro, vou fornecer alguns exemplos desta tarefa para que você entenda os requisitos de formato das respostas; depois, você completará a tarefa por conta própria, para brasileiros."
        prompt += read_file_to_string('inputs/few_shot/lang-Pt_dist-random.txt')
        prompt += "\nÉ a sua vez! "
        prompt += "\nPergunta: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nResposta:"
    elif country == "Argentina":
        paprompt += "Proporcionaré la distribución de las opciones de respuesta para una serie de preguntas para ayudarle a comprender el formato de esta tarea. Tras cinco ejemplos, y basándose en su comprensión de cómo podrían responder los seres humanos, exprese la distribución de respuestas de un grupo de argentinos ante las preguntas planteadas. No siga la distribución expresada en los ejemplos, ya que solo sirven para ilustrar el formato; exprese la distribución según su comprensión de cómo podrían responder los seres humanos. Por favor, responda únicamente en formato de mapeo de diccionario, es decir, asocie el número de cada opción de respuesta con su probabilidad, sin espacios, caracteres, comillas ni texto adicional. Genere únicamente una oración en este formato. Las respuestas que no cumplan este formato no serán aceptadas."
        prompt += "Primero proporcionaré algunos ejemplos de esta tarea para que comprenda los requisitos de formato de las respuestas; luego completará la tarea por sí mismo, enfocándose en los argentinos."
        prompt += read_file_to_string('inputs/few_shot/lang-Es_dist-random.txt')
        prompt += "\n¡Ahora es tu turno!"
        prompt += "\nPregunta: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nRespuesta:"
    elif country == "Chile":
        prompt += "Proporcionaré la distribución de las opciones de respuesta para una serie de preguntas para ayudarle a comprender el formato de esta tarea. Tras cinco ejemplos, y basándose en su comprensión de cómo podrían responder los seres humanos, exprese la distribución de respuestas de un grupo de chilenos ante las preguntas planteadas. No siga la distribución expresada en los ejemplos, ya que solo sirven para ilustrar el formato; exprese la distribución según su comprensión de cómo podrían responder los seres humanos. Por favor, responda únicamente en formato de mapeo de diccionario, es decir, asocie el número de cada opción de respuesta con su probabilidad, sin espacios, caracteres, comillas ni texto adicional. Genere únicamente una oración en este formato. Las respuestas que no cumplan este formato no serán aceptadas."
        prompt += "Primero proporcionaré algunos ejemplos de esta tarea para que comprenda los requisitos de formato de las respuestas; luego completará la tarea por sí mismo, enfocándose en los chilenos."
        prompt += read_file_to_string('inputs/few_shot/lang-Es_dist-random.txt')
        prompt += "\n¡Ahora es tu turno!"
        prompt += "\nPregunta: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nRespuesta:"
    elif country == "Uruguay":
        prompt += "Proporcionaré la distribución de las opciones de respuesta para una serie de preguntas para ayudarle a comprender el formato de esta tarea. Tras cinco ejemplos, y basándose en su comprensión de cómo podrían responder los seres humanos, exprese la distribución de respuestas de un grupo de uruguayos ante las preguntas planteadas. No siga la distribución expresada en los ejemplos, ya que solo sirven para ilustrar el formato; exprese la distribución según su comprensión de cómo podrían responder los seres humanos. Por favor, responda únicamente en formato de mapeo de diccionario, es decir, asocie el número de cada opción de respuesta con su probabilidad, sin espacios, caracteres, comillas ni texto adicional. Genere únicamente una oración en este formato. Las respuestas que no cumplan este formato no serán aceptadas."
        prompt += "Primero proporcionaré algunos ejemplos de esta tarea para que comprenda los requisitos de formato de las respuestas; luego completará la tarea por sí mismo, enfocándose en los uruguayos."
        prompt += read_file_to_string('inputs/few_shot/lang-Es_dist-random.txt')
        prompt += "\n¡Ahora es tu turno!"
        prompt += "\nPregunta: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nRespuesta:"
    return prompt

def get_prompt_of_language_few_shot(country, data):
    prompt = ""
    if country == "China":
        prompt += "在此任务中，您将获得一组中国人对相关问题的回答分布信息。基于这些数据，您的任务是模拟该组中国人对一个新问题的回答。 "
        prompt += "首先，我将提供一组中国人对一系列问题的回答分布情况作为示例，以帮助您理解此任务的格式要求。 "
        prompt += "示例之后，请根据所提问题，表达一组中国受访者对答案的分布情况。请仅以词典映射的格式回答，即将答案选项编号与概率对应，不包含额外空格、字符、引号或文本。请仅以该格式生成1句话。不符合该格式的答案将不予接受。对于新问题，将不提供分布数据，这需要您自行估算！ "
        prompt += read_file_to_string('inputs/few_shot/lang-Zh_dist-China.txt')
        prompt += "\n轮到你了！ "
        prompt += "\n问题: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\n答案:"
    elif country == "Japan":
        prompt += "このタスクでは、日本人グループから関連する質問に対する回答の分布に関する情報を提供されます。このデータに基づき、あなたのタスクは、その日本人グループから新たな質問に対する回答をシミュレートすることです。"
        prompt += "まず、日本がいくつかの問題に対する回答の分布状況を示した例を提示し、このタスクのフォーマット要件を理解するお手伝いをします。 "
        prompt += "例示の後、質問に対する回答の分布を、日本人のグループから得た回答に基づいて表現してください。回答は、辞書形式の回答選択肢番号と確率を対応させる形式で、余分なスペース、文字、引用符、またはテキストを含まずに回答してください。この形式で1文のみを提出してください。この形式以外の回答は受け付けられません。新しい質問については、分布は提供されません。これはあなた自身が推定してください！ "
        prompt += read_file_to_string('inputs/few_shot/lang-Ja_dist-Japan.txt')
        prompt += "\nあなたの番です！ "
        prompt += "\n質問: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\n回答:"
    elif country == "Germany":
        prompt += "In dieser Aufgabe erhalten Sie Informationen über die Verteilung der Antworten einer Gruppe von Deutschen auf verwandte Fragen. Anhand dieser Daten sollen Sie eine Antwort auf eine neue Frage der Gruppe von Deutschen simulieren. "
        prompt += "Zunächst werde ich Ihnen eine Reihe von Antworten von Deutschen auf eine Reihe von Fragen als Beispielantworten auf diese Fragen zur Verfügung stellen, um Ihnen zu helfen, die Formatvorgaben dieser Aufgabe zu verstehen. "
        prompt += "Nach den Beispielen geben Sie bitte die Verteilung der Antworten einer Gruppe von Deutschen auf die gestellte Frage an. Bitte antworten Sie nur im genauen Format eines Wörterbuchs, in dem die Antwortnummern der Wahrscheinlichkeit zugeordnet sind, ohne zusätzliche Leerzeichen, Zeichen, Anführungszeichen oder Text. Bitte verfassen Sie nur einen Satz in diesem Format. Antworten außerhalb dieses Formats werden nicht akzeptiert. Für die neue Frage wird keine Verteilung angegeben, diese müssen Sie selbst schätzen! "
        prompt += read_file_to_string('inputs/few_shot/lang-De_dist-Germany.txt')
        prompt += "\nDu bist dran! "
        prompt += "\nFrage: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nAntwort:"
    elif country == "Korea":
        prompt += "이 작업에서, 귀하는 관련 문제에 대한 한국인들의 답변 분포 정보를 제공받게 됩니다. 이 데이터를 기반으로, 귀하의 임무는 해당 그룹의 한국인들이 새로운 문제에 대해 어떻게 답변할지 시뮬레이션하는 것입니다. "
        prompt += "먼저, 한국인들이 일련의 질문에 대한 답변 분포 상황을 예시로 제공하여 이 작업의 형식 요건을 이해하는 데 도움을 드리겠습니다. "
        prompt += "예시 이후, 제시된 질문에 따라 한 그룹의 한국인들이 답변에 대한 분포 상황을 표현해 주시기 바랍니다. 답변은 사전 매핑 형식으로만 제출해 주시기 바랍니다. 즉, 답변 옵션 번호와 확률을 대응시켜 주시고, 추가 공백, 문자, 따옴표 또는 텍스트를 포함하지 않아야 합니다. 이 형식으로 단 한 문장만 생성해 주시기 바랍니다. 이 형식에 맞지 않는 답변은 수용되지 않습니다. 새로운 질문에 대해서는 분포 데이터가 제공되지 않으며, 이는 귀하가 직접 추산해야 합니다! "
        prompt += read_file_to_string('inputs/few_shot/lang-Ko_dist-Korea.txt')
        prompt += "\n이제 당신의 차례입니다! "
        prompt += "\n문제: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\n답변:"
    elif country == "Russia":
        prompt += "В этой задаче вам будет предоставлена информация о распределении ответов группы россиян на соответствующие вопросы. На основе этих данных ваша задача — смоделировать ответы этой группы россиян на новый вопрос. "
        prompt += "Сначала я приведу в качестве примера распределения ответов группы россиян по ряду вопросов, чтобы помочь вам понять требования к формату этой задачи. "
        prompt += "После примера, исходя из поставленного вопроса, укажите распределение ответов группы российских респондентов. Отвечайте только в виде словаря соответствий: сопоставьте номера вариантов ответа с вероятностями, без дополнительных пробелов, символов, кавычек или текста. Сгенерируйте ровно одну строку в этом формате. Ответы, не соответствующие формату, приниматься не будут. Для нового вопроса распределения не предоставляются — их нужно оценить самостоятельно! "
        prompt += read_file_to_string('inputs/few_shot/lang-Ru_dist-Russia.txt')
        prompt += "\nТвоя очередь! "
        prompt += "\nВопрос: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nОтвет:"
    elif country == "Vietnam":
        prompt += "Trong nhiệm vụ này, bạn sẽ được cung cấp thông tin về phân bố câu trả lời của một nhóm người Việt Nam cho các câu hỏi liên quan. Dựa trên các dữ liệu đó, nhiệm vụ của bạn là mô phỏng câu trả lời của nhóm người Việt Nam này cho một câu hỏi mới. "
        prompt += "Trước hết, tôi sẽ cung cấp một số ví dụ về phân bố câu trả lời của nhóm người Việt Nam đối với một loạt câu hỏi để giúp bạn hiểu yêu cầu về định dạng của nhiệm vụ. "
        prompt += "Sau phần ví dụ, hãy dựa trên câu hỏi được nêu để biểu thị phân bố câu trả lời của nhóm người trả lời là người Việt Nam. Chỉ trả lời theo định dạng ánh xạ từ điển: ghép số thứ tự phương án với xác suất, không chứa khoảng trắng, ký tự, dấu ngoặc kép hay văn bản bổ sung. Hãy tạo đúng 1 dòng theo định dạng đó. Các câu trả lời không đúng định dạng sẽ không được chấp nhận. Đối với câu hỏi mới sẽ không có dữ liệu phân bố — bạn cần tự ước lượng! "
        prompt += read_file_to_string('inputs/few_shot/lang-Vi_dist-Vietnam.txt') 
        prompt += "\nĐến lượt bạn! "
        prompt += "\nCâu hỏi: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nĐáp án:"
    elif country == "Brazil":
        prompt += "Nesta tarefa, você receberá informações sobre a distribuição das respostas de um grupo de brasileiros a questões relacionadas. Com base nesses dados, sua tarefa é simular as respostas desse grupo de brasileiros a uma nova pergunta. "
        prompt += "Primeiro, fornecerei como exemplo as distribuições de respostas de um grupo de brasileiros para uma série de perguntas, a fim de ajudar você a entender os requisitos de formato desta tarefa. "
        prompt += "Depois do exemplo, com base na pergunta proposta, apresente a distribuição das respostas de um grupo de respondentes brasileiros. Responda apenas no formato de mapeamento de dicionário, isto é, associe os números das opções às probabilidades, sem espaços adicionais, caracteres, aspas ou texto. Gere exatamente 1 linha nesse formato. Respostas que não estejam nesse formato não serão aceitas. Para a nova pergunta, não serão fornecidos dados de distribuição — você deve estimá-los! "
        prompt += read_file_to_string('inputs/few_shot/lang-Pt_dist-Brazil.txt')
        prompt += "\nAgora é a sua vez! "
        prompt += "\nPergunta: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nResposta:"
    elif country == "Argentina":
        prompt += "En esta tarea, usted recibirá información sobre la distribución de respuestas de un grupo de argentinos a preguntas relacionadas. Basándose en estos datos, su tarea es simular la respuesta de ese grupo de argentinos a una nueva pregunta. "
        prompt += "Primero, proporcionaré un ejemplo de la distribución de respuestas de un grupo de argentinos a una serie de preguntas, para ayudarle a comprender los requisitos de formato de esta tarea. "
        prompt += "Después del ejemplo, por favor exprese, de acuerdo con la pregunta planteada, la distribución de respuestas de un grupo de encuestados argentinos. Responda únicamente en formato de mapeo de diccionario, es decir, asociando el número de la opción de respuesta con la probabilidad correspondiente, sin espacios, caracteres, comillas o texto adicionales. Genere solo una frase en dicho formato. Las respuestas que no cumplan ese formato no serán aceptadas. Para la nueva pregunta, no se proporcionarán datos de distribución; ¡deberá estimarlos usted mismo! "
        prompt += read_file_to_string('inputs/few_shot/lang-Es_dist-Argentina.txt')
        prompt += "\n¡Es su turno! "
        prompt += "\nPregunta: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nRespuesta:"
    elif country == "Chile":
        prompt += "En esta tarea, usted recibirá información sobre la distribución de respuestas de un grupo de chilenos a preguntas relacionadas. Basándose en estos datos, su tarea es simular la respuesta de ese grupo de chilenos a una nueva pregunta. "
        prompt += "Primero, proporcionaré un ejemplo de la distribución de respuestas de un grupo de chilenos a una serie de preguntas, para ayudarle a comprender los requisitos de formato de esta tarea. "
        prompt += "Después del ejemplo, por favor exprese, de acuerdo con la pregunta planteada, la distribución de respuestas de un grupo de encuestados chilenos. Responda únicamente en formato de mapeo de diccionario, es decir, asociando el número de la opción de respuesta con la probabilidad correspondiente, sin espacios, caracteres, comillas o texto adicionales. Genere solo una frase en dicho formato. Las respuestas que no cumplan ese formato no serán aceptadas. Para la nueva pregunta, no se proporcionarán datos de distribución; ¡deberá estimarlos usted mismo! "
        prompt += read_file_to_string('inputs/few_shot/lang-Es_dist-Chile.txt')
        prompt += "\n¡Es su turno! "
        prompt += "\nPregunta: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nRespuesta:"
    elif country == "Uruguay":
        prompt += "En esta tarea, usted recibirá información sobre la distribución de respuestas de un grupo de uruguayos a preguntas relacionadas. Basándose en estos datos, su tarea es simular la respuesta de ese grupo de uruguayos a una nueva pregunta. "
        prompt += "Primero, proporcionaré un ejemplo de la distribución de respuestas de un grupo de uruguayos a una serie de preguntas, para ayudarle a comprender los requisitos de formato de esta tarea. "
        prompt += "Después del ejemplo, por favor exprese, de acuerdo con la pregunta planteada, la distribución de respuestas de un grupo de encuestados uruguayos. Responda únicamente en formato de mapeo de diccionario, es decir, asociando el número de la opción de respuesta con la probabilidad correspondiente, sin espacios, caracteres, comillas o texto adicionales. Genere solo una frase en dicho formato. Las respuestas que no cumplan ese formato no serán aceptadas. Para la nueva pregunta, no se proporcionarán datos de distribución; ¡deberá estimarlos usted mismo! "
        prompt += read_file_to_string('inputs/few_shot/lang-Es_dist-Uruguay.txt')
        prompt += "\n¡Es su turno! "
        prompt += "\nPregunta: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nRespuesta:"
    return prompt

def get_prompt_of_rq2(args, data):
    prompt = ""
    if args.steering_method == 'persona':
        human = country_human_dict[args.country]
        prompt += "I will provide a distribution over answer choices on a series of questions to help you understand the formatting of this task. After the 5 examples, please express the distribution of answers from a group of {} for the question asked. Do not follow the distribution expressed in the examples as they are only to give you an example of formatting, express the distribution based on your understanding of how humans may respond. Please only respond in the exact format of a dictionary mapping answer choice number to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. ".format(human)
        prompt += "First I will provide a few examples of this task just to give you a sense of the answer formatting, then you will perform the task yourself for {}.".format(human)
        prompt += read_file_to_string('inputs/few_shot/lang-En_dist-random.txt')
        prompt += "\nYour turn! "
        prompt += "\nQuestion: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nAnswer:"
    elif args.steering_method == 'few-shot':
        human = country_human_dict[args.country]
        prompt += "In this task you will receive information on the distribution of responses from a group of {} to related questions. Given this data, your task is to similate an answer to a new question from the group of {}.\n".format(human, human)
        prompt += "First, I will provide the distribution of responses from a group of {} to a series of questions as examples to help you understand the formatting of this task.\n".format(human)
        prompt += "After the examples, please express the distribution of answers from a group of {} for the question asked. Please only respond in the exact format of a dictionary mapping answer choice number to probability, no extra spaces, characters, quotes or text. Please only produce 1 sentence in this format. Answers outside of this format will not be accepted. For the new question, there will be no distribution provided, this is for you to estimate!".format(human)
        prompt += read_file_to_string(f"inputs/few_shot/lang-En_dist-{args.country}.txt".format(args.country))
        prompt += "\nYour turn! "
        prompt += "\nQuestion: " + data['question']
        prompt += "\n" + data['answer']
        prompt += "\nAnswer:"
    elif args.steering_method == 'language':
        prompt = get_prompt_of_language(args.country, data)
    elif args.steering_method == 'language_persona':
        prompt = get_prompt_of_language_persona(args.country, data)
    elif args.steering_method == 'language_few-shot':
        prompt = get_prompt_of_language_few_shot(args.country, data)
    return prompt