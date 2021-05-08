from hmmlearn import hmm
import numpy as np

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.evaluation.generalization import evaluator as generalization_evaluator
from pm4py.evaluation.simplicity import evaluator as simplicity_evaluator
from pm4py.algo.discovery.inductive.parameters import Parameters
from pm4py.objects.process_tree import bottomup as bottomup_util
from pm4py.objects.conversion.process_tree import converter as pt_converter
from constants import *
import constants

import pandas as pd
import csv
import os


def bic_hmmlearn(X, lengths, clusters, trace_dict, df):
    print(activities_column)
    print(constants.start_timestamp)
    file_name = df.file_name
    results_dict = {}
    cluster_range = range(2, clusters + 1)
    # cluster_range = range(2, 4)

    event_df = df.copy()
    event_df[activities_column] = event_df[activities_column].astype(str)
    event_df.rename(
        columns={id_column: 'case:concept:name', date_column: 'time:timestamp', activities_column: 'concept:name'},
        inplace=True)

    log_conversion_parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case',
                                 log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name',
                                 Parameters.START_TIMESTAMP_KEY: start_timestamp,
                                 Parameters.TIMESTAMP_KEY: date_column}

    abstracted_log = log_converter.apply(event_df, parameters=log_conversion_parameters,
                                          variant=log_converter.Variants.TO_EVENT_LOG)
    abstracted_parent = inductive_miner.apply_tree(abstracted_log)

    gviz = pt_visualizer.apply(abstracted_parent,
                               parameters={pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
    pt_visualizer.save(gviz, 'og.png')

    for cluster_num in cluster_range:

        dir_path = os.path.join(os.path.dirname(__file__), 'cluster', file_name + "_c" + str(cluster_num))
        # os.chdir(dir_path)
        # os.mkdir(path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        ind_df = pd.DataFrame([], columns=['Type', 'Net', 'Initial Marking', 'Final Marking'])
        results_dict[cluster_num] = {}
        # results_df = [df, df]
        abstracted_df = df.copy()

        try:
            model = build_cluster_model(cluster_num, X, lengths)
            abstracted_df = predict_cluster(trace_dict, abstracted_df, model)
        except:
            return {}

        clustered_df = abstracted_df.copy()
        projected_df = abstracted_df.copy()
        parent_tree_param = {}

        if cluster_duplicate:

            if constants.start_timestamp is None:
                # global start_timestamp
                constants.start_timestamp = 'start_timestamp'
                abstracted_df[start_timestamp] = abstracted_df[date_column].copy()

            projected_df = abstracted_df.groupby([id_column, 'abstracted'], as_index=False).agg(
                    {date_column: 'max', start_timestamp: 'min'})
            projected_df = pd.merge(projected_df,
                                         clustered_df[[id_column, date_column, 'abstracted', activities_column]],
                                         on=[id_column, date_column, 'abstracted'], how='left')

            projected_df = projected_df.sort_values(by=[id_column, date_column])
            abstracted_df = abstracted_df.sort_values(by=[id_column, date_column])
            parent_tree_param = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case',
                                 log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name',
                                 Parameters.START_TIMESTAMP_KEY: start_timestamp,
                                 Parameters.TIMESTAMP_KEY: date_column}

        elif constants.start_timestamp is not None:
            abstracted_df = abstracted_df.sort_values(by=[id_column, start_timestamp])
            parent_tree_param = {Parameters.START_TIMESTAMP_KEY: start_timestamp}

        projected_df.abstracted = projected_df.abstracted.astype(str)
        projected_df.rename(
            columns={id_column: 'case:concept:name', date_column: 'time:timestamp', 'abstracted': 'concept:name'},
            inplace=True)

        abstracted_df.abstracted = abstracted_df.abstracted.astype(str)
        abstracted_df.rename(
            columns={id_column: 'case:concept:name', date_column: 'time:timestamp', 'abstracted': 'concept:name'},
            inplace=True)


        # define the name of the directory to be created

        dir_path = os.path.join(os.path.dirname(__file__), 'cluster', file_name + "_c" + str(cluster_num))
        #dir_path = cluster_path + file_name + "_c" + str(cluster_num)
        # os.chdir(dir_path)
        # os.mkdir(path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        projected_df.to_csv(os.path.join(dir_path, 'projected' + str(cluster_num) + '.csv'), encoding='utf-8', index=False)
        abstracted_df.to_csv(os.path.join(dir_path, 'cluster' + str(cluster_num) + '.csv'), encoding='utf-8', index=False)

        # results_df.insert(cluster_num, abstracted_trace)

        log_conversion_parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case',
                                     log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name',
                                     Parameters.START_TIMESTAMP_KEY: start_timestamp,
                                     Parameters.TIMESTAMP_KEY: date_column}

        abstracted_log = log_converter.apply(abstracted_df, parameters=log_conversion_parameters,
                                              variant=log_converter.Variants.TO_EVENT_LOG)
        abstracted_parent = inductive_miner.apply_tree(abstracted_log, parameters=parent_tree_param)

        projected_log = log_converter.apply(projected_df, parameters=log_conversion_parameters,
                                              variant=log_converter.Variants.TO_EVENT_LOG)
        projected_tree = inductive_miner.apply_tree(projected_log, parameters=parent_tree_param)

        net, im, fm = pt_converter.apply(projected_tree)

        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case',
                      log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}

        evaluate(event_df, net, im, fm, parameters, file_name + "_c" + str(cluster_num), "projected")

        gviz = pt_visualizer.apply(projected_tree,
                                   parameters={pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
        pt_visualizer.save(gviz, os.path.join(dir_path, 'projected_tree.png'))

        gviz = pt_visualizer.apply(abstracted_parent,
                                   parameters={pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
        pt_visualizer.save(gviz, os.path.join(dir_path, 'abstract_parent.png'))

        i = 0
        while i < cluster_num:
            # ahl_dict[i] = clustered_df[clustered_df['abstracted']==i][activities_column].unique()

            ahl = clustered_df.loc[clustered_df['abstracted'] == i].copy()

            tree_param = {Parameters.ACTIVITY_KEY: activities_column}
            if cluster_duplicate and False:
                ahl = remove_cluster_duplicates(ahl)
                tree_param = {Parameters.START_TIMESTAMP_KEY: start_timestamp,
                              Parameters.ACTIVITY_KEY: activities_column}

            elif constants.start_timestamp is not None:
                tree_param = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case',
                              log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name',
                              Parameters.START_TIMESTAMP_KEY: start_timestamp,
                              Parameters.ACTIVITY_KEY: activities_column}
                # ahl = remove_partial_cluster_duplicates(ahl)

            ahl[activities_column] = ahl[activities_column].astype(str)
            ahl.rename(columns={id_column: 'case:concept:name', date_column: 'time:timestamp'}, inplace=True)

            ahl.to_csv(os.path.join(dir_path, file_name + "_c" + str(cluster_num) + "_" + str(i) + '.csv'),
                       encoding='utf-8', index=False)

            parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case',
                          log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name',
                          Parameters.START_TIMESTAMP_KEY: start_timestamp,
                          Parameters.TIMESTAMP_KEY: date_column}

            log = log_converter.apply(ahl, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
            cluster_tree = inductive_miner.apply_tree(log, parameters=tree_param)

            gviz = pt_visualizer.apply(cluster_tree,
                                       parameters={pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
            pt_visualizer.save(gviz, os.path.join(dir_path, 'ahl' + str(i) + '.png'))

            merge_tree(abstracted_parent, cluster_tree, i)

            # new_row = {'Type': 'AHL'+str(i), 'Net': net, 'Initial Marking': initial_marking, 'Final Marking': final_marking}
            # append row to the dataframe
            # ind_df = ind_df.append(new_row, ignore_index=True)
            i = i + 1

        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case',
                      log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}

        gviz = pt_visualizer.apply(abstracted_parent,
                                   parameters={pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
        pt_visualizer.save(gviz, os.path.join(dir_path, 'merged_parent.png'))

        net, im, fm = pt_converter.apply(abstracted_parent)

        # pt_visualizer.view(gviz)

        results_dict[cluster_num] = evaluate(event_df, net, im, fm, parameters, file_name + "_c" + str(cluster_num), "abstracted")
        # results_dict[cluster_num] = ind_df

    return results_dict


def merge_tree(parent_tree, cluster_tree, i):
    bottomup_nodes = bottomup_util.get_bottomup_nodes(parent_tree)
    for item in bottomup_nodes:
        # print(item._get_label())
        if item._get_label() is not None and str(item._get_label()) == str(float(i)):
            parent = item._get_parent()
            child_list = parent._get_children()
            x = child_list.index(item)
            child_list[x] = cluster_tree
            cluster_tree._set_parent(parent)
            parent._set_children(child_list)


def build_cluster_model(cluster_num, X, lengths):
    model = hmm.GaussianHMM(n_components=cluster_num, covariance_type='full', n_iter=10, verbose=False, tol=0.1)
    # hmm_curr.startprob_ = np.array([0.6, 0.3, 0.1])
    model.fit(X, lengths)
    # print(hmm_curr.monitor_.converged)  # returns True
    print("Cluster: " + str(cluster_num))
    print("Log Score: " + str(model.score(X, lengths)))

    return model


def predict_cluster(trace_dict, abstracted_df, model):
    for key, value in trace_dict.items():
        abstracted_trace = model.predict(np.array(value).reshape(-1, 1))
        idx = abstracted_df[abstracted_df[id_column] == key].index
        abstracted_df.loc[idx, 'abstracted'] = abstracted_trace

    return abstracted_df


def evaluate(event_df, net, initial_marking, final_marking, parameters, file_name, type):
    event_log_model = log_converter.apply(event_df, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    fitness_dict = replay_fitness_evaluator.apply(event_log_model, net, initial_marking, final_marking,
                                                  variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    prec = precision_evaluator.apply(event_log_model, net, initial_marking, final_marking,
                                     variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    gen = generalization_evaluator.apply(event_log_model, net, initial_marking, final_marking)
    simp = simplicity_evaluator.apply(net)

    fitness = fitness_dict.get("perc_fit_traces") / 100
    f1_score = (2 * fitness * prec) / (fitness + prec)
    score = (prec * 0.5) + (((simp + gen + fitness) / 3) * 0.5)

    update_report(fitness, prec, gen, simp, f1_score, score, file_name, type)

    print("Fitness: " + str(fitness))
    print("Precision: " + str(prec))
    print("Generalization: " + str(gen))
    print("Simplicity: " + str(simp))
    print("F1 Score: " + str(f1_score))
    print("Score: " + str(score))
    return score


def update_report(fitness, prec, gen, simp, f1_score, score, file_name, type):
    with open('report.csv', mode='a+') as report:
        report_writer = csv.writer(report, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow([file_name, type, fitness, prec, gen, simp, f1_score, score])


def remove_cluster_duplicates(ahl):
    idx = ahl.groupby([id_column], as_index=False)[date_column].idxmax()
    idx1 = ahl.groupby([id_column], as_index=False)[date_column].idxmin()

    try:
        end = ahl.loc[idx, ]
        begin = ahl.loc[idx1, ]
        ahl = pd.concat([ahl.loc[idx, ], ahl.loc[idx1, ]])
    except:
        ahl = pd.DataFrame(columns=ahl.columns)

    return ahl


def remove_partial_cluster_duplicates(ahl):
    idx = ahl.groupby([id_column, activities_column], as_index=False)[date_column].idxmax()
    # idx1 = ahl.groupby([id_column, activities_column], as_index = False)[date_column].idxmin()

    try:
        ahl = ahl.loc[idx, ]
        # begin = ahl.loc[idx1,]
        # ahl = pd.concat([ahl.loc[idx,],ahl.loc[idx1,]])
    except:
        ahl = pd.DataFrame(columns=ahl.columns)

    return ahl

    # bic_curr = bic_general(hmm_curr.score, free_parameters, X, lengths)
    # bic.append(bic_curr)
    # if bic_curr < lowest_bic:
    #    lowest_bic = bic_curr
    # best_hmm = hmm_curr
    # return (best_hmm, bic, lowest_bic)

    # new_row = {'Type': 'HL', 'Net': net, 'Initial Marking': initial_marking, 'Final Marking': final_marking}
    # append row to the dataframe
    # ind_df = ind_df.append(new_row, ignore_index=True)

    # net, initial_marking, final_marking = inductive_miner.apply(event_log_train, variant= inductive_miner.Variants.IMf)


'''

#        train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7).split(abstracted_df, groups=abstracted_df['case:concept:name']))
#        train = abstracted_df.iloc[train_inds]
#        test = abstracted_df.iloc[test_inds]




def bic_general(likelihood_fn, k, X, lengths):
    """likelihood_fn: Function. Should take as input X and give out   the log likelihood
                  of the data under the fitted model.
           k - int. Number of parameters in the model. The parameter that we are trying to optimize.
                    For HMM it is number of states.
                    For GMM the number of components.
           X - array. Data that been fitted upon.
    """
    bic = np.log(len(X))*k - 2*likelihood_fn(X, lengths)
    return bic



#X1 = np.random.random((25, 1))  # 2-D
#X2 = np.random.random((35, 1))   # must have the same ndim as X1.
#X = np.concatenate([X1, X2])
#lengths = np.array([len(X1), len(X2)])
#best_hmm, bic = bic_hmmlearn(X, lengths)

#print(bic_hmmlearn(X, lengths))

# >>> hmm.GaussianHMM(n_components=2).fit(X, lenghts)
# GaussianHMM(algorithm='viterbi', covariance_type='diag', covars_prior=0.01,
#       covars_weight=1, init_params='stmc', means_prior=0, means_weight=0,
#       min_covar=0.001, n_components=2, n_iter=10, params='stmc',
#       random_state=None, startprob_prior=1.0, tol=0.01, transmat_prior=1.0,
#       verbose=False)


'''

#            temp_first = abstracted_df[abstracted_df[id_column]==key].sort_values(date_column).drop_duplicates(['abstracted'], keep='first')
#            temp_last = abstracted_df[abstracted_df[id_column]==key].sort_values(date_column).drop_duplicates(['abstracted'], keep='last')

#           print(temp_first[[date_column,'start_date_time']])
#           temp_first[[date_column]] = temp_last[[date_column]]
#           print(temp_first[[date_column,'start_date_time']])
#           result = pd.concat([temp_first, temp_last])
#           abstracted_df[abstracted_df[id_column]==key] = temp_first.sort_values(date_column).drop_duplicates()


# Calculate number of free parameters
# free_parameters = for_means + for_covars + for_transmat + for_startprob
# for_means & for_covars = n_features*n_components
# n_features = hmm_curr.n_features
# free_parameters = 2*(n_components*n_features) + n_components*(n_components-1) + (n_components-1)
