import pandas as pd 
import re
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import csv

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

    return student_df, gold_standard_df, student_id

def files_filter(lists, sub_string):
    filtered_list = []
    for file in lists:
        if sub_string in file and 'validation' not in file:
            filtered_list.append(file)
    return filtered_list

def evaluate(student_df, gold_standard_df, student_id):
    grouped_gold_standard_df = gold_standard_df.groupby(gold_standard_df.columns[1])
    total_score = 0
    Maximum_score = 0
    feedback = []
    wrong_predictions = []
    missed_rows = []
    for group_name, group_data in grouped_gold_standard_df:
        # print('Group Name:', group_name)
        # print(group_data)
        # Convert each group to a dictionary
        group_dict = group_data.to_dict(orient='records')
        student_group_prediction = []
        for row in group_dict:
            # print(row)
            # {0: 3001, 1: ' war criminal', 2: 'student', 3: 0.2}

            # read a row from the student's file by the first column value
            student_row = student_df.loc[student_df[0] == row[0]]
            #convert the row to a dictionary
            try:
                student_row_dict = student_row.to_dict(orient='records')[0]
                # print(student_row_dict)
                # {0: 3001, 1: 0.0534579625760864}
                student_group_prediction.append(student_row_dict)
            except:
                # print('Student row not found! Row ID:', row[0])
                missed_rows.append(row[0])
                continue
        # print(group_dict)
        # print(student_group_prediction)
        # Give a rank to each dictionary in the student_group_prediction list based on the value of the second column
        sorted_student_group_prediction = sorted(student_group_prediction, key=lambda x: x[1], reverse=True)
        sorted_gold_standard_group = sorted(group_dict, key=lambda x: x[3], reverse=True)
        # print(sorted_student_group_prediction)
        # print(sorted_gold_standard_group)
        Maximum_score += len(sorted_gold_standard_group)

        feedback_added = False

        if len(sorted_student_group_prediction) == len(sorted_gold_standard_group):
            for i in range(len(sorted_student_group_prediction)):
                if sorted_student_group_prediction[i][0] == sorted_gold_standard_group[i][0]:
                    total_score += 1
                else:
                    if not feedback_added:
                        feedback.append(sorted_gold_standard_group)
                        wrong_predictions.append(sorted_student_group_prediction)
                        feedback_added = True

    return {
        'student_id': student_id,
        'total_score': total_score,
        'Maximum_score': Maximum_score,
        'feedback': feedback,
        'wrong_predictions': wrong_predictions,
        'missed_rows': missed_rows
    }

def write_feedback(result_dict, save_path):
    student_id = result_dict['student_id']
    total_score = result_dict['total_score']
    Maximum_score = result_dict['Maximum_score']
    feedback = result_dict['feedback']
    wrong_predictions = result_dict['wrong_predictions']
    missed_rows = result_dict['missed_rows']

    # write feedback and wrong predictions to a txt file
    feedback_file = save_path
    with open(feedback_file, 'w') as f:
        if student_id == 'Unknown':
            f.write('!Student ID is not found in the file name. Please rename the file to include your student ID')
            f.write('\n')

        f.write('Maxmimum possible Score: '+ str(Maximum_score))
        f.write('\n')
        f.write('Your Score: '+ str(total_score))
        f.write('\n')
        f.write('\n')

        f.write('Feedback:')
        f.write('\n')
        
        for i in range(len(feedback)):
            f.write('Words or phrases involving: '+ feedback[i][0][1])
            f.write('\n')
            f.write('* Correct relative order of similarity scores should be: \n')
            for j in range(len(feedback[i])):
                if j == 0:
                    f.write("    ")
                f.write("{"+str(feedback[i][j][0]))
                f.write(', ')
                f.write(str(feedback[i][j][1]))
                f.write(', ')
                f.write(str(feedback[i][j][2]))
                f.write(', ')
                f.write(str(feedback[i][j][3])+"}")
                if j != len(feedback[i]) - 1:
                    f.write(' > ')
                
            f.write('\n')
            f.write('* Your relative order of similarity scores is: \n')
            for j in range(len(wrong_predictions[i])):
                if j == 0:
                    f.write("    ")
                f.write("{"+str(wrong_predictions[i][j][0]))
                f.write(', ')
                f.write(str(wrong_predictions[i][j][1])+"}")
                if j != len(wrong_predictions[i]) - 1:
                    f.write(' > ')
                else:
                    f.write('\n')
                    f.write('-'*50)
                    f.write('\n')
        
        f.write('\n')
        f.write('Some rows were not found in your submission. Please make sure you have included all the rows in your submission: \n')
        for row in missed_rows:
            f.write('    Row ID: '+str(row) + " is not found in your submission")
            f.write('\n')


def mark_and_record(student_file_path, gold_standard_file_path, task_name):
    student_df, gold_standard_df, student_id = read_data(student_file_path, gold_standard_file_path)
    result_dict = evaluate(student_df, gold_standard_df, student_id)
    save_path = 'feedbacks/'+'student_' +student_id+'_'+task_name+ '_feedback.txt'
    write_feedback(result_dict, save_path)
    # print('Feedback has been written to file: '+ save_path)
    return result_dict


def map_to_normal_distribution(all_results, mean=70, min_value=40, max_value=100, std_dev_scale=3):
    input_list = [result['total_score'] for result in all_results]
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


def draw_raw_score(all_scores, task_name):
    min_score = min(all_scores)
    max_score = max(all_scores)
    n, bins, patches = plt.hist(all_scores, bins=range(int(min_score), int(max_score)+1), density=False, facecolor='g', alpha=0.75)
    plt.xlabel('Raw Score')
    plt.ylabel('Number of Students')
    print(str(len(all_scores)) + " student submissions have been processed for " + task_name)

def draw_mapped_score(mapped_scores):
    max_score = max(mapped_scores)
    min_score = min(mapped_scores)
    n, bins, patches = plt.hist(mapped_scores, bins=range(int(min_score), int(max_score)+1), density=False, facecolor='g', alpha=0.75)
    plt.xlabel('Mapped Score')
    plt.ylabel('Number of Students')

def save_to_csv(mapped_results, file_name):
    keys = mapped_results[0].keys()
    with open(file_name, 'w', newline='\n') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(mapped_results)
    print('Mapped scores have been saved to file: ' + file_name)