import os
import sys
import time
import shutil
import logging
import pyrouge
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=12):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    
    try:
        sns.set(font_scale=4.0)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)

def get_average_length(bert, data): 
    sent_lengths = []
    for item in data:
        encoded_text = bert.tokenize(bert.add_special_token(item['source']))
        sent_lengths.append(len(encoded_text))
    avg_length = sum(sent_lengths)/len(sent_lengths)
    return avg_length

# Test n-gram blocking
def block_ngrams(candidate, prediction, n):
    n_c = _get_ngrams(n, candidate.split())
    for s in prediction:
        n_s = _get_ngrams(n, s.split())
        if len(n_c.intersection(n_s)) > 0:
            return True
    return False

def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n]))
    return ngram_set

def pad(data, pad_id, width=None):
    if not width:  
        width = max([len(d) for d in data])
    
    # Pad until the longest length 
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data

def compute_rouge_score(fold, epoch, save_file, temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding="utf-8")]
    references = [line.strip() for line in open(ref, encoding="utf-8")]
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.makedirs(temp_dir, exist_ok=True)
    tmp_dir = os.path.join(temp_dir, "rouge-score-{}".format(current_time))
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(f"{tmp_dir}/candidate_{fold}", exist_ok=True)
    os.makedirs(f"{tmp_dir}/reference_{fold}", exist_ok=True)

    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(
                f"{tmp_dir}/candidate_{fold}/cand.{i}.txt", "w", encoding="utf-8"
            ) as f:
                f.write(candidates[i].replace("<q>", "\n"))
            with open(
                f"{tmp_dir}/reference_{fold}/ref.{i}.txt", "w", encoding="utf-8"
            ) as f:
                f.write(references[i].replace("<q>", "\n"))
        
        # pyrouge 
        r = pyrouge.Rouge155()
        r.model_dir = f"{tmp_dir}/reference_{fold}/" 
        r.system_dir = f"{tmp_dir}/candidate_{fold}/" 
        r.model_filename_pattern = "ref.#ID#.txt"
        r.system_filename_pattern = "cand.(\d+).txt"
        command = '-e /content/rouge/tools/ROUGE-1.5.5/data -a -b 75 -c 95 -m -n 2'
        rouge_results = r.convert_and_evaluate(rouge_args=command)
        results_dict_rouge = r.output_to_dict(rouge_results)
        
        rouge_1_recall = "{:.2f}".format(float(results_dict_rouge['rouge_1_recall'] * 100))
        rouge_2_recall = "{:.2f}".format(float(results_dict_rouge['rouge_2_recall'] * 100))
        rouge_l_recall = "{:.2f}".format(float(results_dict_rouge['rouge_l_recall'] * 100))
        
        rouge_1_precision = "{:.2f}".format(float(results_dict_rouge['rouge_1_precision'] * 100))
        rouge_2_precision = "{:.2f}".format(float(results_dict_rouge['rouge_2_precision'] * 100))
        rouge_l_precision = "{:.2f}".format(float(results_dict_rouge['rouge_l_precision'] * 100))
        
        rouge_1_f_score = "{:.2f}".format(float(results_dict_rouge['rouge_1_f_score'] * 100))
        rouge_2_f_score = "{:.2f}".format(float(results_dict_rouge['rouge_2_f_score'] * 100))
        rouge_l_f_score = "{:.2f}".format(float(results_dict_rouge['rouge_l_f_score'] * 100))
        results = {}
        
        # Recall
        results['recall-rouge-1'] = rouge_1_recall
        results['recall-rouge-2'] = rouge_2_recall
        results['recall-rouge-l'] = rouge_l_recall
        
        # Precision
        results['precision-rouge-1'] = rouge_1_precision
        results['precision-rouge-2'] = rouge_2_precision
        results['precision-rouge-l'] = rouge_l_precision
        
        # F1-Score
        results['f1-rouge-1'] = rouge_1_f_score
        results['f1-rouge-2'] = rouge_2_f_score
        results['f1-rouge-l'] = rouge_l_f_score
        
        
        # save rouge score at specified file
        corpus_type = str(tmp_dir.split('_')[0])
        evaluation_dir = "final_evaluation_" + corpus_type
        os.makedirs(evaluation_dir, exist_ok=True)
        with open(os.path.join(evaluation_dir, f"{save_file}.txt"), 'w', encoding='utf-8') as f:
            f.write("ROUGE Score\n")
            f.write(rouge_results + '\n\n')
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    
    return results_dict_rouge, results
    