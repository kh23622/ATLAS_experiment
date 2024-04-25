import uproot
import awkward as ak
import vector
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import infofile
import argparse
import pickle
import os
import logging

# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Command line arguments
parser = argparse.ArgumentParser(description='Runs the HZZ analysis on data')
parser.add_argument('--rank', default=0, help='which division node is doing')
args = parser.parse_args()

lumi = 10  # fb-1 for data_A, data_B, data_C, data_D
fraction = 1.0

tuple_path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/"

samples = {
    'data': {'list': ['data_A', 'data_B', 'data_C', 'data_D']},
    r'Background $Z,t\bar{t}$': {'list': ['Zee', 'Zmumu', 'ttbar_lep'], 'color': "#6b59d3"},
    r'Background $ZZ^*$': {'list': ['llll'], 'color': "#ff0000"},
    r'Signal ($m_H$ = 125 GeV)': {'list': ['ggH125_ZZ4lep', 'VBFH125_ZZ4lep', 'WH125_ZZ4lep', 'ZH125_ZZ4lep'],
                                   'color': "#00cdff"}
}

MeV = 0.001
GeV = 1.0

def get_xsec_weight(sample):
    info = infofile.infos[sample]
    xsec_weight = (lumi * 1000 * info["xsec"]) / (info["sumw"] * info["red_eff"])
    return xsec_weight

def calc_weight(xsec_weight, events):
    return (xsec_weight * events.mcWeight * events.scaleFactor_PILEUP * events.scaleFactor_ELE * events.scaleFactor_MUON
            * events.scaleFactor_LepTRIGGER)

def calc_mllll(lep_pt, lep_eta, lep_phi, lep_E):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "E": lep_E})
    return (p4[:, 0] + p4[:, 1] + p4[:, 2] + p4[:, 3]).M * MeV

def cut_lep_charge(lep_charge):
    return lep_charge[:, 0] + lep_charge[:, 1] + lep_charge[:, 2] + lep_charge[:, 3] != 0

def cut_lep_type(lep_type):
    sum_lep_type = lep_type[:, 0] + lep_type[:, 1] + lep_type[:, 2] + lep_type[:, 3]
    return (sum_lep_type != 44) & (sum_lep_type != 48) & (sum_lep_type != 52)

def main(start_index, end_index):
    logging.info('=======================')
    logging.info(f'Processing tasks from {start_index} to {end_index}')
    logging.info('=======================')

    start = time.time()
    data = get_data_from_files(start_index, end_index)
    elapsed = time.time() - start
    logging.info("Time taken: " + str(round(elapsed, 1)) + "s")

    output_directory = '/home/Nischayee/hzz_analysis/worker_hzz/data'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file_path = f'{output_directory}/data_{start_index}_{end_index}.pkl'
    logging.info(f'Saving data to: {output_file_path}')
    with open(output_file_path, 'wb') as d:
        pickle.dump(data, d)
        logging.info(f'data from tasks {start_index} to {end_index} saved at: {output_file_path}')

def get_data_from_files(start_index, end_index):
    data = {}
    for s in samples:
        logging.info(f'Processing {s} samples from {start_index} to {end_index}')
        frames = []
        for val in samples[s]['list']:
            if s == 'data':
                prefix = "Data/"
            else:
                prefix = "MC/mc_" + str(infofile.infos[val]["DSID"]) + "."
            fileString = tuple_path + prefix + val + ".4lep.root"
            temp = read_file(fileString, val, start_index, end_index)
            frames.append(temp)
        data[s] = ak.concatenate(frames)

    return data

def read_file(path, sample, start_index, end_index):
    start = time.time()
    logging.info(f"\tProcessing: {sample}, start_index: {start_index}, end_index: {end_index}")
    data_all = []

    with uproot.open(path + ":mini") as tree:
        numevents = tree.num_entries
        if end_index > numevents:
            end_index = numevents
        for data in tree.iterate(['lep_pt', 'lep_eta', 'lep_phi', 'lep_E', 'lep_charge', 'lep_type',
                                  'mcWeight', 'scaleFactor_PILEUP', 'scaleFactor_ELE', 'scaleFactor_MUON',
                                  'scaleFactor_LepTRIGGER'], library="ak", entry_start=start_index, entry_stop=end_index):

            nIn = len(data)
            if 'data' not in sample:
                xsec_weight = get_xsec_weight(sample)
                data['totalWeight'] = calc_weight(xsec_weight, data)

            data = data[~cut_lep_charge(data.lep_charge)]
            data = data[~cut_lep_type(data.lep_type)]
            data['mllll'] = calc_mllll(data.lep_pt, data.lep_eta, data.lep_phi, data.lep_E)

            nOut = len(data)
            data_all.append(data)
            elapsed = time.time() - start
            logging.info(f"\t\t nIn: {nIn},\t nOut: {nOut}\t in {round(elapsed, 1)}s")

    return ak.concatenate(data_all)

if __name__ == '__main__':
    rank = int(args.rank)
    start_index = rank * 1000
    end_index = start_index + 1000
    main(start_index, end_index)
