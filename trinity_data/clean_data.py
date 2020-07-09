import json
import pickle
import os
import sys
import numpy as np
import pandas as pd
import re

def ProcFig1Data(filepath):
    df = pd.read_csv(filepath, header=None)
    df[['dist','rssi']] = df[0].str.split(expand=True)
    df = df.drop([0], axis=1)
    return df

def ProcFig2Data(filepath):
    df = pd.read_csv(filepath, header=None)
    df[['dist','rssi']] = df[0].str.split(expand=True)
    df = df.drop([0], axis=1)
    return df

def ProcFig3Data(filepath):
    df = pd.read_csv(filepath, header=None)
    df[['dist','rssi']] = df[0].str.split(expand=True)
    df['dist'] = 1
    df = df.drop([0], axis=1)
    return df

def ProcFig4Data(filepath):
    df = pd.read_csv(filepath, header=None)
    df[['dist','rssi']] = df[0].str.split(expand=True)
    df['dist'] = 1
    df = df.drop([0], axis=1)
    return df

def ProcFig5Data(filepath):
    df = pd.read_csv(filepath, header=None)
    df[['dist','rssi']] = df[0].str.split(expand=True)
    df = df.drop([0], axis=1)
    df['dist'] = 1
    return df

def ProcFig7Data(filepath):
    df = pd.read_csv(filepath, header=None)
    df[['dist','rssi']] = df[0].str.split(expand=True)

    substr = re.compile(r'fig7[abcd]').search(filepath).group(0)
    if 'a' in substr:
        df['dist'] = 1/2
    elif 'b' in substr or 'c' in substr:
        df['dist'] = 1
    elif 'd' in substr:
        df['dist'] = 2

    df = df.drop([0], axis=1)
    return df

def ProcFig8Data(filepath):
    df = pd.read_csv(filepath, header=None)
    df[['dist','rssi']] = df[0].str.split(expand=True)
    df = df.drop([0], axis=1)
    df['dist'] = 1
    return df

def ProcFig9Data(filepath):
    df = pd.read_csv(filepath, header=None)
    df[['dist','rssi']] = df[0].str.split(expand=True)

    substr = re.compile(r'fig9[abcd]_').search(filepath).group(0)
    if 'a' in substr or 'c' in substr:
        df['dist'] = 4/5
    if 'b' in substr or 'd' in substr:
        df['dist'] = 1
    df = df.drop([0], axis=1)
    return df

def ProcFig13Data(filepath):
    df = pd.read_csv(filepath, header=None)
    df[['dist','rssi']] = df[0].str.split(expand=True)
    df = df.drop([0], axis=1)

    substr = re.compile(r'fig13[ab]_').search(filepath).group(0)
    if 'a' in substr:
        df['dist'] = 1/2
    if 'b' in substr:
        df['dist'] = 2

    return df

def ProcFig14Data(filepath):
    df = pd.read_csv(filepath, header=None)
    df[['dist','rssi']] = df[0].str.split(expand=True)
    df = df.drop([0], axis=1)

    substr = re.compile(r'fig14[abc]_').search(filepath).group(0)
    if 'a' in substr:
        df['dist'] = 2
    if 'b' in substr or 'c' in substr:
        df['dist'] = 4
    return df


if __name__ == '__main__':
    """
    Aggregates selected data provided by Doug Leith
    from Trinity College. Writes to specified
    output_file

    1) outdoor RSSI vs distance measurements (0.5-4m)
    2) Indoor RSSI vs distance measurements (0.5-4m)
    3) Person orientation (1m)
    4) Person around the device (front-back-left-right) (1m)
    5) Phone in purse (1m)
    7) Carrying phone, walking near each other (0.5-2m)
    8) Handset orientation to each other (1m)
    9) People around table (0.8, 1m)
    13) People in supermarket (0.5, 2m)
    14) In line outside supermarket (2, 4m)

    ble_params: [d_ref  (reference distance in m), power_ref (received power at the reference distance), path_loss_exp (path loss exponent), stdev_power (standard deviation of received power in dB), threshold (signals below this value cannot be received)]
    Outside: [1, -63.5664, 2.3022, 2.8101, -100]
    Inside: [1, -62.9190, 2.3158, 3.4418, -100]
    Train: [1, -60.4517, 1.3640, 5.054, -100]

    """
    path = os.getcwd() + "/"
    files = [path+item for item in os.listdir(path) if (".csv" in item and "fig" in item)]
    net_df = None

    dist_val = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    for fpath in files:
        temp_df = None
        if re.compile(r'fig1[a]_').search(fpath):
            if 'huawei' in fpath:
                continue
            if 'samsung' in fpath:
                continue
            temp_df = ProcFig1Data(fpath)
        # if re.compile(r'fig2[a]_').search(fpath):
        #     temp_df = ProcFig2Data(fpath)
        # if re.compile(r'fig3_').search(fpath):
        #     temp_df = ProcFig3Data(fpath)
        # if re.compile(r'fig4_').search(fpath):
        #     temp_df = ProcFig4Data(fpath)
        # if re.compile(r'fig5_').search(fpath):
        #     temp_df = ProcFig5Data(fpath)
        # if re.compile(r'fig7[abcd]_').search(fpath):
        #     temp_df = ProcFig7Data(fpath)
        # if re.compile(r'fig8_').search(fpath):
        #     temp_df = ProcFig8Data(fpath)
        # if re.compile(r'fig9[abcd]_').search(fpath):
        #     temp_df = ProcFig9Data(fpath)
        # if re.compile(r'fig13[ab]_').search(fpath):
        #     temp_df = ProcFig13Data(fpath)
        # if re.compile(r'fig14[abc]_').search(fpath):
        #     temp_df = ProcFig14Data(fpath)

        if temp_df is not None:
            if net_df is None:
                net_df = temp_df
            else:
                net_df = pd.concat([net_df, temp_df], ignore_index=True)

    net_df['dist'] = net_df['dist'].astype(float)
    net_df = net_df.sort_values(by=['dist']).reset_index(drop=True)

    for dv in dist_val:
        print(net_df[net_df['dist']==dv]['rssi'].astype(float).std())

    # print(net_df[net_df['dist']==1]['rssi'].astype(float))

    # output_file = path+"aggregate_data.csv"
    # net_df.to_csv(output_file, index=False)


