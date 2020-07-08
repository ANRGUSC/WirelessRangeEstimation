
import re
import time
import json
import pickle
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from scipy.special import ndtr


def MakeSummaryTables(files, snl_approaches):

    max_err_df = pd.DataFrame()
    max_percent_err_df = pd.DataFrame()
    avg_err_df = pd.DataFrame()
    avg_percent_err_df = pd.DataFrame()
    true_pos_df = pd.DataFrame()
    false_pos_df = pd.DataFrame()
    runtime_df = pd.DataFrame()

    for filepath in files:
        trials_dict = pd.read_excel(filepath, sheet_name=None, index_col=0)
        max_err_df = max_err_df.append(trials_dict['max_error'], sort=False)
        max_percent_err_df = max_percent_err_df.append(trials_dict['max_percent_error'], sort=False)
        avg_err_df = avg_err_df.append(trials_dict['avg_error'], sort=False)
        avg_percent_err_df = avg_percent_err_df.append(trials_dict['avg_percent_error'], sort=False)
        true_pos_df = true_pos_df.append(trials_dict['true_positive'], sort=False)
        false_pos_df = false_pos_df.append(trials_dict['false_positive'], sort=False)
        runtime_df = runtime_df.append(trials_dict['runtime'], sort=False)

    max_err_df.reset_index(inplace=True)
    avg_err_df.reset_index(inplace=True)
    max_percent_err_df.reset_index(inplace=True)
    avg_percent_err_df.reset_index(inplace=True)
    true_pos_df.reset_index(inplace=True)
    false_pos_df.reset_index(inplace=True)
    runtime_df.reset_index(inplace=True)

    approaches = list(max_err_df.columns)[1:]
    approaches.sort()
    print(approaches)

    table = pd.DataFrame()
    for app in approaches:
        table[app] = 



