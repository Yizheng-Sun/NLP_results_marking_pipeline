import pandas as pd
import re
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import norm

def read_data(student_file_path, gold_standard_file_path):
    eight_digit_number_regex = r'\d{8}'
    student_id = re.findall(eight_digit_number_regex, student_file_path)
    if student_id:
        student_id = student_id[0]
    else:
        student_id = 'Unknown'

    student_df = pd.read_csv(student_file_path, header=None)
    # print(student_df.head())
    gold_standard_file = gold_standard_file_path
    gold_standard_df = pd.read_csv(gold_standard_file, header=None)
    gold_standard_df = gold_standard_df.drop([1,2,3], axis=1)
    gold_standard_df = gold_standard_df.iloc[1:]

    return student_df, gold_standard_df, student_id

def match_student_and_gold_standard(student_df, gold_standard_df):
    gold_standard_reformatted = []
    student_results_reformatted = []
    missed_rows = []
    student_df_copy = student_df.copy()

    for index, row in gold_standard_df.iterrows():
        # rename column names from 0 to n
        row = row.reset_index(drop=True)
        row_found = False
        row_id = row[0]
        row_labels = [int(row[i]) for i in range(1, len(row))]
        gold_standard_reformatted.append(row_labels)

        for index, student_row in student_df_copy.iterrows():
            if student_row[0] == row_id:
                student_row_labels = [int(student_row[i]) for i in range(1, len(student_row))]
                student_results_reformatted.append(student_row_labels)
                row_found = True
                student_df_copy.drop(index, inplace=True)
                break
        if not row_found:
            missed_rows.append(row_id)
            student_results_reformatted.append([0 if label == 1 else 1 for label in row_labels])

    return gold_standard_reformatted, student_results_reformatted, missed_rows  

def evaluate(gold_standard_reformatted, student_results_reformatted):
    f1_averaged = f1_score(gold_standard_reformatted, student_results_reformatted, average='weighted')

    f1_unaveraged = f1_score(gold_standard_reformatted, student_results_reformatted, average=None)

    return f1_averaged, f1_unaveraged

def map_to_normal_distribution(all_results, mean=70, min_value=40, max_value=100, std_dev_scale=3):
    input_list = [result['f1_score'] for result in all_results]
    mapped_results = []
    # Step 1: Normalize the input list to [0, 1]
    min_input = min(input_list)
    max_input = max(input_list)
    normalized_list = [(x - min_input) / (max_input - min_input) for x in input_list]

    # Step 2: Map the normalized values to the desired range [40, 100]
    scaled_list = [min_value + x * (max_value - min_value) for x in normalized_list]

    # Step 3: Adjust to a normal distribution centered around the mean value
    normal_dist = norm(loc=mean, scale=(max_value - mean) / std_dev_scale)
    mapped_list = [normal_dist.ppf(x) for x in normalized_list]

    # Ensure values are within the desired range
    mapped_list = np.clip(mapped_list, min_value, max_value)
    for score, result_dict in zip(mapped_list, all_results):
        temp_dict = {}
        temp_dict['student_id'] = result_dict['student_id']
        temp_dict['mapped_score'] = score
        mapped_results.append(temp_dict)

    return mapped_list, mapped_results

def write_feedback(student_id, f1_score, f1_score_unaveraged, missed_rows, task_name):
    with open(f'feedbacks/student_{student_id}_{task_name}_feedback.txt', 'w') as f:
        f.write(f'Your F1 Score: {f1_score}\n')
        f.write(f'Your F1 Score For each class:\n')
        for i, score in enumerate(f1_score_unaveraged):
            f.write(f'   Class {i}: {score}\n')

        f.write('************************************\n')
        if len(missed_rows) > 0:
            f.write(f'The following rows are not found in your submission: {missed_rows}\n')
            for row in missed_rows:
                f.write(f'{row}\n')

def mark_and_record(student_file_path, gold_standard_file_path, task_name):
    student_df, gold_standard_df, student_id = read_data(student_file_path, gold_standard_file_path)
    gold_standard_reformatted, student_results_reformatted, missed_rows = match_student_and_gold_standard(student_df, gold_standard_df)
    f1_score, f1_unaveraged = evaluate(gold_standard_reformatted, student_results_reformatted)

    write_feedback(student_id, f1_score, f1_unaveraged, missed_rows, task_name)
    return {'student_id': student_id, 'f1_score': f1_score}