import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import time
import random
from tqdm import tqdm
import functions as fn

# Initialize time
start_time = time.time()

# Read the data
url = "https://raw.githubusercontent.com/chen700564/RGB/master/data/en.json"
data = fn.process_json(url)
data = random.sample(data, 20)
queries = [item["query"] for item in data]
answers = [item["answer"][0] for item in data]

# Set up models
ext_model_1 = "timpal0l/mdeberta-v3-base-squad2"
ext_model_2 = "deepset/xlm-roberta-large-squad2"
ext_model_3 = "google-bert/bert-base-multilingual-cased"
ext_model_4 = "mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp"
ext_model_5 = "deepset/roberta-base-squad2"
ext_model_6 = "distilbert/distilbert-base-cased-distilled-squad"
ext_model_7 = "deepset/electra-base-squad2"

models = [ext_model_1]
separator = " <|> "

for model in models:

    tokenizer = AutoTokenizer.from_pretrained(model)
    loaded_model = AutoModelForQuestionAnswering.from_pretrained(model)
    qa_pipeline = pipeline("question-answering", model=loaded_model, tokenizer=tokenizer)

    # Process data
    results = []
    for query, positive_context, negative_context, answer in tqdm(zip(queries, [item["positive"] for item in data], [item["negative"] for item in data], answers), total=len(queries)):
        combined_context = positive_context + negative_context
        result = {
            'Query': query,
            'Correct Answer': answer,
        }
        result.update({'Positive ' + k: v for k, v in fn.process_context(positive_context, qa_pipeline, query, separator).items()})
        result.update({'Negative ' + k: v for k, v in fn.process_context(negative_context, qa_pipeline, query, separator).items()})
        result.update({'PosNeg ' + k: v for k, v in fn.process_context(combined_context, qa_pipeline, query, separator).items()})
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df['Jaccard Positive'] = results_df.apply(lambda row: fn.apply_jaccard(row, 'Positive Predicted Answer', 'Correct Answer'), axis=1)
    results_df['Jaccard Negative'] = results_df.apply(lambda row: fn.apply_jaccard(row, 'Negative Predicted Answer', 'Correct Answer'), axis=1)
    results_df['Jaccard PosNeg'] = results_df.apply(lambda row: fn.apply_jaccard(row, 'PosNeg Predicted Answer', 'Correct Answer'), axis=1)
    results_df['Cosine Positive'] = results_df.apply(lambda row: fn.apply_cosine(row, 'Positive Predicted Answer', 'Correct Answer'), axis=1)
    results_df['Cosine Negative'] = results_df.apply(lambda row: fn.apply_cosine(row, 'Negative Predicted Answer', 'Correct Answer'), axis=1)
    results_df['Cosine PosNeg'] = results_df.apply(lambda row: fn.apply_cosine(row, 'PosNeg Predicted Answer', 'Correct Answer'), axis=1)
    results_df['EM Positive'] = results_df.apply(lambda row: fn.apply_exact_match(row, 'Positive Predicted Answer', 'Correct Answer'), axis=1)
    results_df['EM Negative'] = results_df.apply(lambda row: fn.apply_exact_match(row, 'Negative Predicted Answer', 'Correct Answer'), axis=1)
    results_df['EM PosNeg'] = results_df.apply(lambda row: fn.apply_exact_match(row, 'PosNeg Predicted Answer', 'Correct Answer'), axis=1)

    # Save results
    results_df.to_excel(f"results/{model.split('/')[1]}.xlsx", index=False)

end_time = time.time()
print(f"Total execution time: {(end_time - start_time) / 60} minutes.")
