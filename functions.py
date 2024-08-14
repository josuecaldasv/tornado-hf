import requests
import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
import openpyxl
import evaluate
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import os
import ast
import tiktoken
import random
# import nltk


model_st = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
stop_words_english = set(stopwords.words('english'))
exact_match_metric = evaluate.load("exact_match")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

# def check_and_download(resource):
#     try:
#         nltk.data.find(f'tokenizers/{resource}')
#     except LookupError:
#         nltk.download(resource)

# check_and_download('punkt')
# check_and_download('stopwords')

    
def safe_eval(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        if isinstance(x, str):
            return re.findall(r"'(.*?)'", x)
        else:
            return x 


def process_json(url):
    """
    Objective:
        - Fetches a JSON data from the specified URL and processes it to extract individual JSON objects.
          This function is useful for handling concatenated JSON strings that are not properly arrayed.
    
    Input:
        - url: str - The URL from which the JSON data is to be fetched.
    
    Output:
        - data: list - A list of dictionaries, where each dictionary is a parsed JSON object from the fetched data.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        json_strings = re.split(r'\}\s*\n\s*\{', response.text)

        data = []
        for i, json_str in enumerate(json_strings):
            if i != 0:
                json_str = '{' + json_str
            if i != len(json_strings) - 1:
                json_str += '}'
            obj = json.loads(json_str)
            if 'answer' in obj:
                if not all(isinstance(el, list) for el in obj['answer']):
                    obj['answer'] = [obj['answer']]

            data.append(obj)

        return data
    except requests.RequestException as e:
        print("Request Error:", e)
        return []
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        return []


def normalize_answer(text):
    """Lower text and remove punctuation, stop_words and extra whitespace."""
    if isinstance(text, float):
        text = str(text)
    text_without_stop_words = " ".join([word for word in text.split() if word.lower() not in stop_words_english])
    text_white_space_fix = " ".join(text_without_stop_words.split())
    text_without_punctuation = "".join(ch for ch in text_white_space_fix if ch not in string.punctuation)
    text_lower = text_without_punctuation.lower()
    return text_lower.split()



def jaccard_similarity_formula(set1, set2):
    """Calculate the Jaccard similarity between two sets."""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0: 
        print('-------Error: Union is empty-------')
        print(f'Intersection: {intersection}, Union: {union}')
        return 0.0 
    return len(intersection) / len(union)



def calculate_jaccard(predicted, correct_answers):
    normalized_predicted = set(normalize_answer(predicted))
    max_similarity_index = 0
    for answer in correct_answers:
        normalized_answer = set(normalize_answer(answer))
        similarity_index = jaccard_similarity_formula(normalized_predicted, normalized_answer)
        if similarity_index > max_similarity_index:
            max_similarity_index = similarity_index
    return max_similarity_index


def apply_jaccard(row, pred, true):
    """Calcular la similaridad usando la función calculate_jaccard."""
    return calculate_jaccard(row[pred], row[true])


def calculate_exact_match(predicted, correct_answers):
    max_start_match = 0
    predicted_strip = predicted.strip()
    for answer in correct_answers:
        result = exact_match_metric.compute(references=[answer], predictions=[predicted_strip], ignore_case=True, ignore_punctuation=True)
        start_match = result["exact_match"]
        if start_match > max_start_match:
            max_start_match = start_match
    return max_start_match


def calculate_exact_match_2v(predicted, correct_answers):
    """Evaluar si la respuesta predicha coincide exactamente con alguna de las respuestas correctas."""
    normalized_predicted = set(normalize_answer(predicted))
    match = 0  # Iniciar con match = 0
    for answer in correct_answers:
        normalized_answer = set(normalize_answer(answer))
        if all(word in normalized_predicted for word in normalized_answer):
            match = 1
            break  # Salir del bucle si se encuentra una coincidencia
    return match

def apply_exact_match_2v(row, pred, true):
    """Calcular la similaridad usando la función calculate_exact_match_2v."""
    return calculate_exact_match_2v(row[pred], row[true])


def apply_exact_match(row, pred, true):
    """Calcular la similaridad usando la función calculate_em."""
    return calculate_exact_match(row[pred], row[true])

def calculate_cosine(predicted, correct_answers):
    embeddings_pred = model_st.encode(predicted, convert_to_tensor=True)
    for answer in correct_answers:
        embeddings_true = model_st.encode(answer, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embeddings_pred, embeddings_true)
        return cosine_similarity.item()

def apply_cosine(row, pred, true):
    """Calcular la similaridad usando la función calculate_cosine."""
    return calculate_cosine(row[pred], row[true])


# def f1_score(prediction, ground_truth):
#     prediction_tokens = normalize_answer(str(prediction))
#     ground_truth_tokens = normalize_answer(str(ground_truth))
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction)
    ground_truth_tokens = normalize_answer(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def max_f1_score(prediction, ground_truths):
    max_f1 = 0.0
    for gt in ground_truths:
        f1 = f1_score(prediction, gt)
        if f1 > max_f1:
            max_f1 = f1
    return max_f1

# def max_f1_score(prediction, ground_truths):
#     return max(f1_score(prediction, gt) for gt in ground_truths)

def calculate_exact_match(predicted, correct_answers):
    max_start_match = 0
    for answer in correct_answers:
        result = exact_match_metric.compute(references=[answer], predictions=[predicted], ignore_case=True, ignore_punctuation=True)
        start_match = result["exact_match"]
        if start_match > max_start_match:
            max_start_match = start_match
    return max_start_match

def rouge(prediction, correct_answers):
    results = rouge_metric.compute(references=[correct_answers], predictions=[prediction])
    return results["rouge1"]

def bleu(prediction, correct_answers):
    results = bleu_metric.compute(references=[correct_answers], predictions=[prediction])
    return results["bleu"]

# def process_context(context, qa_pipeline, query, separator, max_tokens=None):
#     enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
#     context_concat = separator.join(context)
#     context_tokens = enc.encode(context_concat)
#     limited_tokens = context_tokens[:max_tokens] if max_tokens else context_tokens
#     context_concat_limited = enc.decode(limited_tokens)
#     results = qa_pipeline(question=query, context=context_concat_limited)
#     start_idx = results['start']
#     end_idx = results['end']
#     interval = f"{start_idx} - {end_idx}"
#     document_indexes = []
#     current_index = 0
#     for element in context:
#         start = current_index
#         end = current_index + len(element)
#         document_indexes.append((start, end))
#         current_index = end + len(separator) 

#     document_index = next((i for i, (start, end) in enumerate(document_indexes) if start <= start_idx < end), len(context) - 1)
#     document = context[document_index]
#     return {
#         'Predicted Answer': str(results['answer']).strip(),
#         'Appended Context': str(context_concat),
#         'Context Interval': str(interval),
#         'Document Index': str(document_index),
#         'Document': str(document)
#     }


def process_context(context, qa_pipeline, query, separator):
    context_concat = separator.join(context)
    results = qa_pipeline(question=query, context=context_concat)
    start_idx = results['start']
    end_idx = results['end']
    interval = f"{start_idx} - {end_idx}"
    document_indexes = []
    current_index = 0
    for element in context:
        start = current_index
        end = current_index + len(element)
        document_indexes.append((start, end))
        current_index = end + len(separator)
    document_index = next((i for i, (start, end) in enumerate(document_indexes) if start <= start_idx < end), len(context) - 1)
    document = context[document_index]
    return {
        'Predicted Answer': str(results['answer']).strip(),
        'Appended Context': str(context_concat),
        'Context Interval': str(interval),
        'Document Index': str(document_index),
        'Document': str(document)
    }


def wrap_text_and_add(label, context_results):
    """
    Args:
        label: The label to identify the block of text.
        context_results: A dictionary containing the results for different noise levels.
    """
    yield f"- {label}:"
    for noise_level in ['Noise_0', 'Noise_25', 'Noise_50', 'Noise_75', 'Noise_100']:
        result = context_results.get(noise_level, {})
        answer = result.get('Predicted Answer', 'N/A')
        text = result.get('Appended Context', '')
        match = result.get('EM', False)
        jaccard = result.get('Jaccard', 0.0)
        cosine = result.get('Cosine', 0.0)
        
        formatted_text = " ".join(text.split())  # Limpiar y unir texto
        yield f"  - {noise_level}:"
        yield f"    - Answer              : {answer}"
        yield "    - Threshold           : 0.8"
        yield "    - Source              :"
        start = 0
        max_width = 100
        while start < len(formatted_text):
            end = min(start + max_width, len(formatted_text))
            if end < len(formatted_text):
                end = formatted_text.rfind(' ', start, end)
            if end == -1 or end <= start:  # Ajuste para evitar bucles infinitos
                end = start + max_width
            yield f"                          {formatted_text[start:end]}"
            start = end + 1
        yield f"    - Jaccard Index V.    : {jaccard:.2f}"
        yield f"    - Cosine Similarity V.: {cosine:.2f}"
        yield f"    - Match (EM)          : {'Yes' if match else 'No'}"
        yield f"    - Match (Jaccard)     : {'Yes' if jaccard > 0.8 else 'No'}"
        yield f"    - Match (Cosine)      : {'Yes' if cosine > 0.8 else 'No'}"



def wrap_answers(label, correct_answers):
    """
    Args:
        label: The label to identify the block of answers.
        correct_answers: A list of correct answers (strings).
    """
    yield f"{label}:"
    if isinstance(correct_answers, str) and correct_answers.startswith("[") and correct_answers.endswith("]"):
        correct_answers = eval(correct_answers)
    formatted_text = ", ".join(correct_answers) if isinstance(correct_answers, list) else correct_answers
    
    start = 0
    max_width = 90
    while start < len(formatted_text):
        end = min(start + max_width, len(formatted_text))
        if end < len(formatted_text):
            end = formatted_text.rfind(',', start, end)
        if end == -1 or end <= start:  # Ajuste para evitar bucles infinitos
            end = start + max_width
        yield f"                          {formatted_text[start:end]}"
        start = end + 2


def format_results(data):
    for index, row in data.iterrows():
        yield "=" * 120
        yield f"Question {index + 1}              : {row['Query']}"
        for line in wrap_answers("Correct Answers         :  ", row['Correct Answer']):
            yield line
        yield "-" * 120

        for noise_level in ['Noise_0', 'Noise_25', 'Noise_50', 'Noise_75', 'Noise_100']:
            context_results = {
                'Predicted Answer': row[f'{noise_level} Predicted Answer'],
                'Appended Context': row[f'{noise_level} Appended Context'],
                'EM': row[f'EM {noise_level}'],
                'Jaccard': row[f'Jaccard {noise_level}'],
                'Cosine': row[f'Cosine {noise_level}'],
            }
            yield "-" * 120
            for line in wrap_text_and_add(f"Prediction ({noise_level})", context_results):
                yield line
        yield "=" * 120
        yield "\n"



def read_excel_in_chunks(filename, cols_to_use, chunk_size=1000):
    workbook = openpyxl.load_workbook(filename, read_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(min_row=1, values_only=True)
    headers = next(rows)  # Asume que la primera fila es el encabezado
    col_indices = [headers.index(col) for col in cols_to_use]

    def get_row(row):
        return [row[idx] for idx in col_indices]

    data = []
    for row in rows:
        data.append(get_row(row))
        if len(data) >= chunk_size:
            yield pd.DataFrame(data, columns=cols_to_use)
            data = []
    if data:
        yield pd.DataFrame(data, columns=cols_to_use)

def read_json_in_chunks(filename, cols_to_use, chunk_size=1000):
    with open(filename, 'r') as file:
        chunk = []
        for line in file:
            chunk.append(json.loads(line))
            if len(chunk) == chunk_size:
                yield pd.DataFrame(chunk, columns=cols_to_use)
                chunk = []
        if chunk:
            yield pd.DataFrame(chunk, columns=cols_to_use)

# Lista de columnas adaptadas para los diferentes niveles de ruido
cols_to_use = [ 
    'Query', 
    'Correct Answer', 
    'Noise_0 Predicted Answer', 'Noise_0 Appended Context', 'EM Noise_0', 'EM - 2V Noise_0', 'Cosine Noise_0', 'Jaccard Noise_0',
    'Noise_20 Predicted Answer', 'Noise_20 Appended Context', 'EM Noise_20', 'EM - 2V Noise_20', 'Cosine Noise_20', 'Jaccard Noise_20',
    'Noise_40 Predicted Answer', 'Noise_40 Appended Context', 'EM Noise_40', 'EM - 2V Noise_40', 'Cosine Noise_40', 'Jaccard Noise_40',
    'Noise_60 Predicted Answer', 'Noise_60 Appended Context', 'EM Noise_60', 'EM - 2V Noise_60', 'Cosine Noise_60', 'Jaccard Noise_60',
    'Noise_80 Predicted Answer', 'Noise_80 Appended Context', 'EM Noise_80', 'EM - 2V Noise_80', 'Cosine Noise_80', 'Jaccard Noise_80',
    'Noise_100 Predicted Answer', 'Noise_100 Appended Context', 'EM Noise_100', 'EM - 2V Noise_100', 'Cosine Noise_100', 'Jaccard Noise_100'
]


def compute_metrics(input_file, threshold):
    # Leer el archivo JSON
    df = pd.read_json(input_file, orient='records', lines=True)
    df['Correct Answer'] = df['Correct Answer'].apply(safe_eval)
    
    f1_scores = {f'Noise_{i}': [] for i in [0, 20, 40, 60, 80, 100]}
    rouge_scores = {f'Noise_{i}': [] for i in [0, 20, 40, 60, 80, 100]}
    bleu_scores = {f'Noise_{i}': [] for i in [0, 20, 40, 60, 80, 100]}
    
    em_scores = {f'Noise_{i}': df[f'EM Noise_{i}'].tolist() for i in [0, 20, 40, 60, 80, 100]}
    em_scores_2v = {f'Noise_{i}': df[f'EM - 2V Noise_{i}'].tolist() for i in [0, 20, 40, 60, 80, 100]}
    em_cosine_scores = {f'Noise_{i}': df[f'Cosine Noise_{i}'].apply(lambda x: 1 if x >= threshold else 0).tolist() for i in [0, 20, 40, 60, 80, 100]}
    em_jaccard_scores = {f'Noise_{i}': df[f'Jaccard Noise_{i}'].apply(lambda x: 1 if x >= threshold else 0).tolist() for i in [0, 20, 40, 60, 80, 100]}

    for index, row in df.iterrows():
        correct_answer = row['Correct Answer']
        
        for i in [0, 20, 40, 60, 80, 100]:
            noise_level = f'Noise_{i}'
            
            f1 = max_f1_score(row[f'{noise_level} Predicted Answer'], correct_answer)
            f1_scores[noise_level].append(f1)

            rouge_score = rouge(row[f'{noise_level} Predicted Answer'], correct_answer)
            rouge_scores[noise_level].append(rouge_score)

            bleu_score = bleu(row[f'{noise_level} Predicted Answer'], correct_answer)
            bleu_scores[noise_level].append(bleu_score)
    
    avg_f1 = {noise_level: sum(f1_scores[noise_level]) / len(f1_scores[noise_level]) if f1_scores[noise_level] else 0 for noise_level in f1_scores}
    avg_em = {noise_level: sum(em_scores[noise_level]) / len(em_scores[noise_level]) if em_scores[noise_level] else 0 for noise_level in em_scores}
    avg_em_2v = {noise_level: sum(em_scores_2v[noise_level]) / len(em_scores_2v[noise_level]) if em_scores_2v[noise_level] else 0 for noise_level in em_scores_2v}
    avg_em_cosine = {noise_level: sum(em_cosine_scores[noise_level]) / len(em_cosine_scores[noise_level]) if em_cosine_scores[noise_level] else 0 for noise_level in em_cosine_scores}
    avg_em_jaccard = {noise_level: sum(em_jaccard_scores[noise_level]) / len(em_jaccard_scores[noise_level]) if em_jaccard_scores[noise_level] else 0 for noise_level in em_jaccard_scores}
    avg_rouge = {noise_level: sum(rouge_scores[noise_level]) / len(rouge_scores[noise_level]) if rouge_scores[noise_level] else 0 for noise_level in rouge_scores}
    avg_bleu = {noise_level: sum(bleu_scores[noise_level]) / len(bleu_scores[noise_level]) if bleu_scores[noise_level] else 0 for noise_level in bleu_scores}
    
    result_data = {
        'Metric': [
            'F1', 
            'EM - String',
            'EM - 2V',
            f'EM - Cosine (threshold = {threshold})',
            f'EM - Jaccard (threshold = {threshold})',
            'RougeL',
            'Bleu'
        ]
    }
    
    for noise_level in ['Noise_0', 'Noise_20', 'Noise_40', 'Noise_60', 'Noise_80', 'Noise_100']:
        result_data[noise_level] = [
            avg_f1[noise_level], 
            avg_em[noise_level], 
            avg_em_2v[noise_level], 
            avg_em_cosine[noise_level], 
            avg_em_jaccard[noise_level], 
            avg_rouge[noise_level], 
            avg_bleu[noise_level]
        ]
    
    result_df = pd.DataFrame(result_data)
    return result_df


# def compute_metrics(input_file, threshold):
#     df = pd.read_excel(input_file)
#     df['Correct Answer'] = df['Correct Answer'].apply(ast.literal_eval)
    
#     f1_scores_positive = []
#     f1_scores_negative = []
#     f1_scores_posneg = []
#     rouge_scores_positive = []
#     rouge_scores_negative = []
#     rouge_scores_posneg = []
#     bleu_scores_positive = []
#     bleu_scores_negative = []
#     bleu_scores_posneg = []

#     em_scores_positive = df['EM Positive'].tolist()
#     em_scores_negative = df['EM Negative'].tolist()
#     em_scores_posneg = df['EM PosNeg'].tolist()
    
#     em_cosine_positive = df['Cosine Positive'].apply(lambda x: 1 if x >= threshold else 0).tolist()
#     em_cosine_negative = df['Cosine Negative'].apply(lambda x: 1 if x >= threshold else 0).tolist()
#     em_cosine_posneg = df['Cosine PosNeg'].apply(lambda x: 1 if x >= threshold else 0).tolist()
    
#     em_jaccard_positive = df['Jaccard Positive'].apply(lambda x: 1 if x >= threshold else 0).tolist()
#     em_jaccard_negative = df['Jaccard Negative'].apply(lambda x: 1 if x >= threshold else 0).tolist()
#     em_jaccard_posneg = df['Jaccard PosNeg'].apply(lambda x: 1 if x >= threshold else 0).tolist()

#     for index, row in df.iterrows():
#         correct_answer = row['Correct Answer']
        
#         f1_positive = max_f1_score(row['Positive Predicted Answer'], correct_answer)
#         f1_scores_positive.append(f1_positive)
        
#         f1_negative = max_f1_score(row['Negative Predicted Answer'], correct_answer)
#         f1_scores_negative.append(f1_negative)
        
#         f1_posneg = max_f1_score(row['PosNeg Predicted Answer'], correct_answer)
#         f1_scores_posneg.append(f1_posneg)

#         rouge_positive = rouge(row['Positive Predicted Answer'], correct_answer)
#         rouge_scores_positive.append(rouge_positive)

#         rouge_negative = rouge(row['Negative Predicted Answer'], correct_answer)
#         rouge_scores_negative.append(rouge_negative)

#         rouge_posneg = rouge(row['PosNeg Predicted Answer'], correct_answer)
#         rouge_scores_posneg.append(rouge_posneg)

#         bleu_positive = bleu(row['Positive Predicted Answer'], correct_answer)
#         bleu_scores_positive.append(bleu_positive)

#         bleu_negative = bleu(row['Negative Predicted Answer'], correct_answer)
#         bleu_scores_negative.append(bleu_negative)

#         bleu_posneg = bleu(row['PosNeg Predicted Answer'], correct_answer)
#         bleu_scores_posneg.append(bleu_posneg)
    
#     avg_f1_positive = sum(f1_scores_positive) / len(f1_scores_positive) if f1_scores_positive else 0
#     avg_f1_negative = sum(f1_scores_negative) / len(f1_scores_negative) if f1_scores_negative else 0
#     avg_f1_posneg = sum(f1_scores_posneg) / len(f1_scores_posneg) if f1_scores_posneg else 0
    
#     avg_em_positive = sum(em_scores_positive) / len(em_scores_positive) if em_scores_positive else 0
#     avg_em_negative = sum(em_scores_negative) / len(em_scores_negative) if em_scores_negative else 0
#     avg_em_posneg = sum(em_scores_posneg) / len(em_scores_posneg) if em_scores_posneg else 0
    
#     avg_em_cosine_positive = sum(em_cosine_positive) / len(em_cosine_positive) if em_cosine_positive else 0
#     avg_em_cosine_negative = sum(em_cosine_negative) / len(em_cosine_negative) if em_cosine_negative else 0
#     avg_em_cosine_posneg = sum(em_cosine_posneg) / len(em_cosine_posneg) if em_cosine_posneg else 0
    
#     avg_em_jaccard_positive = sum(em_jaccard_positive) / len(em_jaccard_positive) if em_jaccard_positive else 0
#     avg_em_jaccard_negative = sum(em_jaccard_negative) / len(em_jaccard_negative) if em_jaccard_negative else 0
#     avg_em_jaccard_posneg = sum(em_jaccard_posneg) / len(em_jaccard_posneg) if em_jaccard_posneg else 0

#     avg_rouge_positive = sum(rouge_scores_positive) / len(rouge_scores_positive) if rouge_scores_positive else 0
#     avg_rouge_negative = sum(rouge_scores_negative) / len(rouge_scores_negative) if rouge_scores_negative else 0
#     avg_rouge_posneg = sum(rouge_scores_posneg) / len(rouge_scores_posneg) if rouge_scores_posneg else 0

#     avg_bleu_positive = sum(bleu_scores_positive) / len(bleu_scores_positive) if bleu_scores_positive else 0
#     avg_bleu_negative = sum(bleu_scores_negative) / len(bleu_scores_negative) if bleu_scores_negative else 0
#     avg_bleu_posneg = sum(bleu_scores_posneg) / len(bleu_scores_posneg) if bleu_scores_posneg else 0
    
#     result_data = {
#         'Metric': [
#             'F1', 
#             'EM - String',
#             f'EM - Cosine (threshold = {threshold})',
#             f'EM - Jaccard (threshold = {threshold})',
#             'RougeL',
#             'Bleu'
#         ],
#         'Positive': [
#             avg_f1_positive, avg_em_positive,
#             avg_em_cosine_positive, avg_em_jaccard_positive,
#             avg_rouge_positive, avg_bleu_positive
#         ],
#         'Negative': [
#             avg_f1_negative, avg_em_negative,
#             avg_em_cosine_negative, avg_em_jaccard_negative,
#             avg_rouge_negative, avg_bleu_negative
#         ],
#         'Posnegative': [
#             avg_f1_posneg, avg_em_posneg,
#             avg_em_cosine_posneg, avg_em_jaccard_posneg,
#             avg_rouge_posneg, avg_bleu_posneg
#         ]
#     }
#     result_df = pd.DataFrame(result_data)
#     return result_df


def create_mixed_context(positive_context, negative_context, noise_level, max_total_tokens, separator=" <|> "):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    total_elements = len(positive_context) + len(negative_context)
    
    # Cálculo del número de elementos positivos y negativos deseados
    num_negative = int(total_elements * noise_level)
    num_positive = total_elements - num_negative
    
    # Limitar la selección a la cantidad disponible en cada lista
    positive_sample = random.sample(positive_context, min(num_positive, len(positive_context)))
    negative_sample = random.sample(negative_context, min(num_negative, len(negative_context)))

    # Combinar y limitar por el número de tokens
    combined_context = positive_sample + negative_sample
    random.shuffle(combined_context)  # Barajar los elementos seleccionados
    
    # Concatenar contextos con el separador
    context_concat = separator.join(combined_context)
    context_tokens = enc.encode(context_concat)
    
    # Limitar el número de tokens
    limited_tokens = context_tokens[:max_total_tokens]
    context_concat_limited = enc.decode(limited_tokens)
    
    # Retornar el contexto final
    final_combined_context = context_concat_limited.split(separator)
    random.shuffle(final_combined_context)  # Aleatorizar el contexto final

    return final_combined_context

