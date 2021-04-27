import pm4py

#dataframe = pm4py.read_csv("/home/anand/Documents/study/Thesis - Event Abstraction/data/bpi_2013/data.csv", sep=';', quotechar=None, encoding=None, nrows=None, timest_format="YYYY-MM-DDTHH:MM:SS+HH:MM")

import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


log_csv = pd.read_csv("/home/anand/Documents/study/Thesis - Event Abstraction/data/bpi_2013/data.csv", sep=';')
#log_csv['Status'] = log_csv['Problem Status'] + ' ' + log_csv['Problem Sub Status']
#log_csv = log_csv.drop(['Problem Status','Problem Sub Status'])

cols = ['Problem Status','Problem Sub Status']
newcol = ['+'.join(i) for i in log_csv[cols].astype(str).values]
log_csv = log_csv.assign(Status=newcol).drop(cols, 1)
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
log_csv = log_csv.sort_values('Problem Change Date+Time')

print(log_csv)

parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'Problem Number'}
event_log = log_converter.apply(log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
xes_exporter.apply(event_log, '/home/anand/Documents/study/Thesis - Event Abstraction/data/bpi_2013/output.xes')

#dataframe = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
#dataframe.to_csv('/home/anand/Documents/study/Thesis - Event Abstraction/data/bpi_2013/output.csv')

import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner

log = xes_importer.apply(os.path.join("tests","input_data","/home/anand/Documents/study/Thesis - Event Abstraction/data/bpi_2013/BPI Challenge 2013, closed problems.xes"))
net, initial_marking, final_marking = inductive_miner.apply(log)

dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
dataframe.to_csv('/home/anand/Documents/study/Thesis - Event Abstraction/data/bpi_2013/output.csv')

from pm4py.visualization.petrinet import visualizer as pn_visualizer
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)