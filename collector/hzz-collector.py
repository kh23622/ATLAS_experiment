import argparse
import logging
import os
import pickle
import time
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import awkward as ak

# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Command line arguments
parser = argparse.ArgumentParser(description='Plot data from distributed tasks.')
parser.add_argument('--data_directory', default='/home/Nischayee/hzz_analysis/worker_hzz/data', help='Path to the directory containing distributed task data')
args = parser.parse_args()

#tuple_path = "Input/4lep/" # local
tuple_path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/" # web address

samples = {

    'data': {
        'list' : ['data_A','data_B','data_C','data_D'],
    },

    r'Background $Z,t\bar{t}$' : { # Z + ttbar
        'list' : ['Zee','Zmumu','ttbar_lep'],
        'color' : "#6b59d3" # purple
    },

    r'Background $ZZ^*$' : { # ZZ
        'list' : ['llll'],
        'color' : "#ff0000" # red
    },

    r'Signal ($m_H$ = 125 GeV)' : { # H -> ZZ -> llll
        'list' : ['ggH125_ZZ4lep','VBFH125_ZZ4lep','WH125_ZZ4lep','ZH125_ZZ4lep'],
        'color' : "#00cdff" # light blue
    },

}
MeV = 0.001
GeV = 1.0


# Function to plot data
def plot_data(data):
    xmin = 80 * GeV
    xmax = 250 * GeV
    step_size = 5 * GeV

    bin_edges = np.arange(start=xmin, stop=xmax+step_size, step=step_size)
    bin_centres = np.arange(start=xmin+step_size/2, stop=xmax+step_size/2, step=step_size)

    data_x = np.zeros(len(bin_edges) - 1)
    data_x_errors = np.zeros(len(bin_edges) - 1)

    for key in data.keys():
        if key == 'data':
            data_x, _ = np.histogram(ak.to_numpy(data[key]['mllll']), bins=bin_edges)
            data_x_errors = np.sqrt(data_x)
        elif key.startswith('Signal'):
            signal_x = ak.to_numpy(data[key]['mllll'])
            signal_weights = ak.to_numpy(data[key].totalWeight)
            signal_color = samples[key]['color']
            plt.hist(signal_x, bins=bin_edges, weights=signal_weights, color=signal_color, label=key)
        else:
            mc_x = ak.to_numpy(data[key]['mllll'])
            mc_weights = ak.to_numpy(data[key].totalWeight)
            mc_color = samples[key]['color']
            plt.hist(mc_x, bins=bin_edges, weights=mc_weights, color=mc_color, label=key)

    plt.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors, fmt='ko', label='Data')

    plt.xlabel(r'4-lepton invariant mass $\mathrm{m_{4l}}$ [GeV]')
    plt.ylabel('Events / '+str(step_size)+' GeV')
    plt.xlim(left=xmin, right=xmax)
    plt.ylim(bottom=0)
    plt.minorticks_on()
    plt.grid(which='both')
    plt.legend(frameon=False)
    plt.savefig('data/graph.png')
    plt.show()


if __name__ == '__main__':
    # Wait for data files with a timeout
    timeout = 60  # Timeout in seconds
    start_time = time.time()
    while time.time() - start_time < timeout:
        if glob.glob(os.path.join(args.data_directory, 'data_*.pkl')):
            break
        time.sleep(1)
    else:
        logging.error("Timeout: Data files not found within specified time.")
        exit(1)

    # Load and combine data from distributed task files
    ak_list = []
    for pkl_file_path in glob.glob(os.path.join(args.data_directory, 'data_*.pkl')):
        with open(pkl_file_path, 'rb') as file:
            ak_list.append(pickle.load(file))

    result_dict = {}
    for key in ak_list[0][0].keys():
        arrays = [d[0][key] for d in ak_list]
        concatenated_array = ak.concatenate(arrays, axis=0)
        result_dict[key] = concatenated_array

    # Plot the combined data
    plot_data(result_dict)
