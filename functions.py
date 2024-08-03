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
    """ Calcular el coeficiente de Jaccard (Índice de Similariad) entre dos conjuntos. """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
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

def process_context(context, qa_pipeline, query, separator):
    context_concat = separator.join(context)
    results = qa_pipeline(question=query, context=context_concat)
    start_idx = results['start']
    end_idx = results['end']
    interval = f"{start_idx} - {end_idx}"

    document_indexes = []
    current_index = 0
    # Calculamos los índices de inicio y fin para cada documento
    for element in context:
        start = current_index
        end = current_index + len(element)
        document_indexes.append((start, end))
        current_index = end + len(separator)  # Sumamos el tamaño del separador al índice actual

    # Determinamos el índice del documento donde se encuentra la respuesta
    document_index = next((i for i, (start, end) in enumerate(document_indexes) if start <= start_idx < end), len(context) - 1)
    document = context[document_index]

    return {
        'Predicted Answer': str(results['answer']).strip(),
        'Appended Context': str(context_concat),
        'Context Interval': str(interval),
        'Document Index': str(document_index),
        'Document': str(document)
    }


def wrap_text_and_add(label, text, answer, match, jaccard, cosine):   
    formatted_text = " ".join(text.split())  # Limpiar y unir texto
    yield f"- {label}:"
    yield f"  - Answer              : {answer}"
    yield "  - Threshold           : 0.8"
    yield "  - Source              :"
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
    yield f"  - Jaccard Index V.    : {jaccard:.2f}"
    yield f"  - Cosine Similarity V.: {cosine:.2f}"
    yield f"  - Match (EM)          : {'Yes' if match else 'No'}"
    yield f"  - Match (Jaccard)     : {'Yes' if jaccard > 0.8 else 'No'}"
    yield f"  - Match (Cosine)      : {'Yes' if cosine > 0.8 else 'No'}"


def wrap_answers(label, answers):
    yield f"{label}"
    if isinstance(answers, str) and answers.startswith("[") and answers.endswith("]"):
        answers = eval(answers)
    formatted_text = ", ".join(answers) if isinstance(answers, list) else answers
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
        for line in wrap_answers("Correct Answers         :  ",  row['Correct Answer']):
            yield line
        yield "-" * 120

        for line in wrap_text_and_add("Prediction (only positive)", row['Positive Document'], row['Positive Predicted Answer'], row['EM Positive'], row['Jaccard Positive'], row['Cosine Positive']):
            yield line
        yield "-" * 120

        for line in wrap_text_and_add("Prediction (only negative)", row['Negative Document'], row['Negative Predicted Answer'], row['EM Negative'], row['Jaccard Negative'], row['Cosine Negative']):
            yield line
        yield "-" * 120

        for line in wrap_text_and_add("Prediction (both positive and negative)", row['PosNeg Document'], row['PosNeg Predicted Answer'], row['EM PosNeg'], row['Jaccard PosNeg'], row['Cosine PosNeg']):
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


cols_to_use = [ 'Query', 'Correct Answer', 
                'Positive Document', 'Positive Predicted Answer', 'EM Positive', 'Cosine Positive', 'Jaccard Positive', 
                'Negative Document', 'Negative Predicted Answer', 'EM Negative', 'Cosine Negative', 'Jaccard Negative', 
                'PosNeg Document', 'PosNeg Predicted Answer', 'EM PosNeg', 'Cosine PosNeg', 'Jaccard PosNeg']


def compute_metrics(input_file, threshold):
    # Leer el archivo JSON
    df = pd.read_json(input_file, orient='records', lines=True)
    df['Correct Answer'] = df['Correct Answer'].apply(ast.literal_eval)
    
    f1_scores_positive = []
    f1_scores_negative = []
    f1_scores_posneg = []
    rouge_scores_positive = []
    rouge_scores_negative = []
    rouge_scores_posneg = []
    bleu_scores_positive = []
    bleu_scores_negative = []
    bleu_scores_posneg = []

    em_scores_positive = df['EM Positive'].tolist()
    em_scores_negative = df['EM Negative'].tolist()
    em_scores_posneg = df['EM PosNeg'].tolist()

    em_scores_positive_2v = df['EM Positive - 2V'].tolist()
    em_scores_negative_2v = df['EM Negative - 2V'].tolist()
    em_scores_posneg_2v = df['EM PosNeg - 2V'].tolist()
    
    em_cosine_positive = df['Cosine Positive'].apply(lambda x: 1 if x >= threshold else 0).tolist()
    em_cosine_negative = df['Cosine Negative'].apply(lambda x: 1 if x >= threshold else 0).tolist()
    em_cosine_posneg = df['Cosine PosNeg'].apply(lambda x: 1 if x >= threshold else 0).tolist()
    
    em_jaccard_positive = df['Jaccard Positive'].apply(lambda x: 1 if x >= threshold else 0).tolist()
    em_jaccard_negative = df['Jaccard Negative'].apply(lambda x: 1 if x >= threshold else 0).tolist()
    em_jaccard_posneg = df['Jaccard PosNeg'].apply(lambda x: 1 if x >= threshold else 0).tolist()

    for index, row in df.iterrows():
        correct_answer = row['Correct Answer']
        
        f1_positive = max_f1_score(row['Positive Predicted Answer'], correct_answer)
        f1_scores_positive.append(f1_positive)
        
        f1_negative = max_f1_score(row['Negative Predicted Answer'], correct_answer)
        f1_scores_negative.append(f1_negative)
        
        f1_posneg = max_f1_score(row['PosNeg Predicted Answer'], correct_answer)
        f1_scores_posneg.append(f1_posneg)

        rouge_positive = rouge(row['Positive Predicted Answer'], correct_answer)
        rouge_scores_positive.append(rouge_positive)

        rouge_negative = rouge(row['Negative Predicted Answer'], correct_answer)
        rouge_scores_negative.append(rouge_negative)

        rouge_posneg = rouge(row['PosNeg Predicted Answer'], correct_answer)
        rouge_scores_posneg.append(rouge_posneg)

        bleu_positive = bleu(row['Positive Predicted Answer'], correct_answer)
        bleu_scores_positive.append(bleu_positive)

        bleu_negative = bleu(row['Negative Predicted Answer'], correct_answer)
        bleu_scores_negative.append(bleu_negative)

        bleu_posneg = bleu(row['PosNeg Predicted Answer'], correct_answer)
        bleu_scores_posneg.append(bleu_posneg)
    
    avg_f1_positive = sum(f1_scores_positive) / len(f1_scores_positive) if f1_scores_positive else 0
    avg_f1_negative = sum(f1_scores_negative) / len(f1_scores_negative) if f1_scores_negative else 0
    avg_f1_posneg = sum(f1_scores_posneg) / len(f1_scores_posneg) if f1_scores_posneg else 0
    
    avg_em_positive = sum(em_scores_positive) / len(em_scores_positive) if em_scores_positive else 0
    avg_em_negative = sum(em_scores_negative) / len(em_scores_negative) if em_scores_negative else 0
    avg_em_posneg = sum(em_scores_posneg) / len(em_scores_posneg) if em_scores_posneg else 0

    avg_em_positive_2v = sum(em_scores_positive_2v) / len(em_scores_positive_2v) if em_scores_positive_2v else 0
    avg_em_negative_2v = sum(em_scores_negative_2v) / len(em_scores_negative_2v) if em_scores_negative_2v else 0
    avg_em_posneg_2v = sum(em_scores_posneg_2v) / len(em_scores_posneg_2v) if em_scores_posneg_2v else 0
    
    avg_em_cosine_positive = sum(em_cosine_positive) / len(em_cosine_positive) if em_cosine_positive else 0
    avg_em_cosine_negative = sum(em_cosine_negative) / len(em_cosine_negative) if em_cosine_negative else 0
    avg_em_cosine_posneg = sum(em_cosine_posneg) / len(em_cosine_posneg) if em_cosine_posneg else 0
    
    avg_em_jaccard_positive = sum(em_jaccard_positive) / len(em_jaccard_positive) if em_jaccard_positive else 0
    avg_em_jaccard_negative = sum(em_jaccard_negative) / len(em_jaccard_negative) if em_jaccard_negative else 0
    avg_em_jaccard_posneg = sum(em_jaccard_posneg) / len(em_jaccard_posneg) if em_jaccard_posneg else 0

    avg_rouge_positive = sum(rouge_scores_positive) / len(rouge_scores_positive) if rouge_scores_positive else 0
    avg_rouge_negative = sum(rouge_scores_negative) / len(rouge_scores_negative) if rouge_scores_negative else 0
    avg_rouge_posneg = sum(rouge_scores_posneg) / len(rouge_scores_posneg) if rouge_scores_posneg else 0

    avg_bleu_positive = sum(bleu_scores_positive) / len(bleu_scores_positive) if bleu_scores_positive else 0
    avg_bleu_negative = sum(bleu_scores_negative) / len(bleu_scores_negative) if bleu_scores_negative else 0
    avg_bleu_posneg = sum(bleu_scores_posneg) / len(bleu_scores_posneg) if bleu_scores_posneg else 0
    
    result_data = {
        'Metric': [
            'F1', 
            'EM - String',
            'EM - String (2V)',
            f'EM - Cosine (threshold = {threshold})',
            f'EM - Jaccard (threshold = {threshold})',
            'RougeL',
            'Bleu'
        ],
        'Positive': [
            avg_f1_positive, avg_em_positive, avg_em_positive_2v,
            avg_em_cosine_positive, avg_em_jaccard_positive,
            avg_rouge_positive, avg_bleu_positive
        ],
        'Negative': [
            avg_f1_negative, avg_em_negative, avg_em_negative_2v,
            avg_em_cosine_negative, avg_em_jaccard_negative,
            avg_rouge_negative, avg_bleu_negative
        ],
        'Posnegative': [
            avg_f1_posneg, avg_em_posneg, avg_em_posneg_2v,
            avg_em_cosine_posneg, avg_em_jaccard_posneg,
            avg_rouge_posneg, avg_bleu_posneg
        ]
    }
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