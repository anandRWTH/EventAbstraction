import pandas as pd
import numpy as np
import hmm

from constants import *
import shutil
import os
import json


def preprocess_eventdata(df):
    print(df.info())

    df[date_column] = pd.to_datetime(df[date_column])

    if start_timestamp is None:
        df = df[[id_column, activities_column, date_column]]  # 1;
    else:
        df = df[[id_column, activities_column, date_column, start_timestamp]]  # 1;
    event_list = df[activities_column].unique()
    df[activities_column] = df[activities_column].astype('category')
    df[[activities_column]] = df[[activities_column]].apply(lambda x: x.cat.codes)
    print(np.array(event_list))

    return len(event_list), df


def build_model_params(df):
    cases = df[id_column].unique()

    # trace_count = df[id_column].unique().size

    trace_dict = {}
    observations_list = []
    lengths_list = []

    for case in cases:
        t = df[df[id_column] == case][[activities_column]].values
        flat_list = [item for sublist in t for item in sublist]
        trace_dict[case] = flat_list
        observations_list.extend(flat_list)
        lengths_list.append(len(flat_list))

    print(np.array(lengths_list))
    # print(np.array(trace_dict['1-740866708']).reshape(-1, 1))
    return np.vstack(observations_list), np.array(lengths_list), trace_dict


# X - m, lengths - m , clusters, trace_dict - m, df

def find_copy_csv_file(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    csv_list = []
    for name in filenames:
        if name.endswith(suffix):
            shutil.copy(os.path.join(path_to_dir, name), data_path)
            name = os.path.splitext(os.path.basename(name))[0]
            csv_list.append(name)
    return csv_list


def build_cluster_hierarchy(df, csv_file):
    csv_list = []
    output = {}
    threshold_condition = True
    threshold_value = 0.01
    dir_path = os.getcwd()
    file_name = os.path.splitext(os.path.basename(csv_file.name))[0]
    csv_list.append(file_name)

    output["null"] = file_name

    while threshold_condition:

        if len(df) > 1:
            print(csv_file.name)
            clusters, df = preprocess_eventdata(df)
            X, lengths, trace_dict = build_model_params(df)

            file_name = os.path.splitext(os.path.basename(csv_file.name))[0]
            df.file_name = file_name

            scores = hmm.bic_hmmlearn(X, lengths, clusters, trace_dict, df)
            # scores = {}
            # scores = dict([(2, 0.48), (3, 0.21), (4, 0.20)])
            if scores:
                cluster_max = max(scores, key=scores.get)
                if scores[cluster_max] > threshold_value:
                    src_path = cluster_path + file_name + "_c" + str(cluster_max)
                    temp_list = find_copy_csv_file(src_path)
                    print(temp_list) #['og_c2_1', 'og_c2_0']
                    output[file_name] = temp_list
                    csv_list = csv_list + temp_list
        else:
            file_name = os.path.splitext(os.path.basename(csv_file.name))[0]

        csv_list.remove(file_name)
        print(output)

        if csv_list:
            csv_file = open(data_path + csv_list[0] + ".csv")
            df = pd.read_csv(csv_file, engine='python', delimiter=',')
            df.rename(columns={'case:concept:name': id_column, 'time:timestamp': date_column}, inplace=True)
            df.drop('abstracted', axis=1, inplace=True)
        else:
            threshold_condition = False
    tree = get_nodes("og", "null", output)
    return json.dumps(tree)

def get_nodes(node, parent, data):

    d = {}
    d['name'] = node
    d['display'] = node
    d['parent'] = parent

    if node in data:
        children = data[node]
        d['folder'] = d['name'] + '_c' + str(len(children))
        d['children'] = [get_nodes(child, node, data) for child in children]

    return d


def build_hierarchical_json():
    csv_file = open(og_data_file)
    df = pd.read_csv(csv_file, engine='python', delimiter=',')

    # df = pd.read_csv(og_data_file, engine='python', delimiter=';')
    # df[activities_column] = df['STAGE'] + " + " + df['ACTIVITY'];

    return build_cluster_hierarchy(df, csv_file)

"""if __name__ == "__main__":

    csv_file = open(og_data_file)
    df = pd.read_csv(csv_file, engine='python', delimiter=',')

    # df = pd.read_csv(og_data_file, engine='python', delimiter=';')
    # df[activities_column] = df['STAGE'] + " + " + df['ACTIVITY'];

    if cluster_hierarchy:
        build_cluster_hierarchy(df, csv_file)
    else:
        clusters, df = preprocess_eventdata(df)
        X, lengths, trace_dict = build_model_params(df)
        file_name = os.path.splitext(os.path.basename(csv_file.name))[0]
        df.file_name = file_name
        hmm.bic_hmmlearn(X, lengths, clusters, trace_dict, df)"""