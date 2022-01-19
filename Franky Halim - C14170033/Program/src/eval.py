# For evaluate training and validation
import os,sys
import pyrouge
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser(description='Evaluation Training and Validation Indonesian News Summarization')
parser.add_argument('--fold',type=int,default=1, help='fold for evaluate ROUGE')
parser.add_argument('--epochs',type=int,default=4, help='number of epochs')
parser.add_argument('--saved_name', type=str, default='indolem-indobert-base-uncased_use-token-type-ids_2_1_1e-05_0.1_8_4', help='path to saved file') 
args = parser.parse_args()

# Plot function
def plot_rouge(max_epoch, rouge_train, rouge_val, title):
    epoch = range(1, max_epoch + 1)
    plt.figure()
    plt.plot(epoch, rouge_train, label='Training')
    plt.plot(epoch, rouge_val, label='Validation')
    plt.title(title)
    plt.ylabel('F1-Score')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0.7, 1.0])
    plt.show()

def compute_rouge_score(fold, epochs, save_file, temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding="utf-8")]
    references = [line.strip() for line in open(ref, encoding="utf-8")]
    assert len(candidates) == len(references)

    cnt = len(candidates)
    os.makedirs(temp_dir, exist_ok=True)
    tmp_dir = os.path.join(temp_dir, "rouge-score")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(f"{tmp_dir}/candidate_{fold}", exist_ok=True)
    os.makedirs(f"{tmp_dir}/reference_{fold}", exist_ok=True)


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
    evaluation_dir = "final_evaluation_" + temp_dir
    os.makedirs(evaluation_dir, exist_ok=True)
    with open(os.path.join(evaluation_dir, f"{save_file}.txt"), 'w', encoding='utf-8') as f:
        f.write("ROUGE Score\n")
        f.write(rouge_results + '\n\n')
    
    return results_dict_rouge, results

train_rouge = []
val_rouge = []
for epoch in range(1, args.epochs + 1):
    # Training
    temp_dir = "train"
    cand = args.saved_name + "/train_pred_" + str(epoch) + ".txt" 
    ref = args.saved_name + "/train_gold_" + str(epoch) + ".txt"
    train_raw_rouge, train_rouge_score = compute_rouge_score(args.fold, args.epochs, args.saved_name, temp_dir, cand, ref)
    train_rouge.append(train_raw_rouge)
    
    # Validation
    temp_dir = "val"
    cand = args.saved_name + "/val_pred_" + str(epoch) + ".txt" 
    ref = args.saved_name + "/val_gold_" + str(epoch) + ".txt"
    val_raw_rouge, val_rouge_score = compute_rouge_score(args.fold, args.epochs, args.saved_name, temp_dir, cand, ref)
    val_rouge.append(val_raw_rouge)
 
# ROUGE-1
rouge_train = [r['rouge_1_f_score'] for r in train_rouge]
rouge_val = [r['rouge_1_f_score'] for r in val_rouge]
#plot_rouge(args.epochs, rouge_train, rouge_val, "ROUGE-1")
print("Use this value to plot\n\n")
print("ROUGE-1")
print("Training ROUGE-1: ",rouge_train)
print("Validation ROUGE-1: ", rouge_val)
print("\n")

# ROUGE-2
rouge_train = [r['rouge_2_f_score'] for r in train_rouge]
rouge_val = [r['rouge_2_f_score'] for r in val_rouge]
#plot_rouge(args.epochs, rouge_train, rouge_val, "ROUGE-2")
print("ROUGE-2")
print("Training ROUGE-2: ",rouge_train)
print("Validation ROUGE-2: ", rouge_val)
print("\n")

# ROUGE-L
rouge_train = [r['rouge_l_f_score'] for r in train_rouge]
rouge_val = [r['rouge_l_f_score'] for r in val_rouge]
#plot_rouge(args.epochs, rouge_train, rouge_val, "ROUGE-L")
print("ROUGE-L")
print("Training ROUGE-L: ",rouge_train)
print("Validation ROUGE-L: ", rouge_val)
print("\n")
   
    
