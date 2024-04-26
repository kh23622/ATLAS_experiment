import argparse
import logging
import os
import pickle
import time
import awkward as ak
import infofile
import uproot
import vector

# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Command line arguments
parser = argparse.ArgumentParser(description='Split data processing into multiple segments and save start/end points.')
parser.add_argument('--number_workers', default=1, help='Number of workers for data processing')
args = parser.parse_args()


def count_file(path, sample):
    """
    Counts events in a file.

    Args:
        path (str): Path to the input file.
        sample (str): Name of the sample.

    Returns:
        int: Number of events in the file.
    """
    start = time.time()  # start the clock
    logging.info(f"\tCounting: {sample}")  # print which sample is being processed
    # open the tree called mini using a context manager (will automatically close files/resources)
    with uproot.open(path + ":mini") as tree:
        numevents = tree.num_entries  # number of events
        count = numevents  # number of events in this batch

        elapsed = time.time() - start  # time taken to process
        logging.info(f"\t\t Count: {count},\t in {round(elapsed, 1)}s")  # number of counts

    return count  # return number of events

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

def get_data_from_files():
    """
    Process all files and get counts for each sample.

    Returns:
        dict: Dictionary containing counts for each sample.
    """
    counts = {}  # define empty dictionary to hold counts
    for s in samples:  # loop over samples
        dict_counts = {}
        logging.info(f'Counting {s} samples')  # print which sample
        for val in samples[s]['list']:  # loop over each file
            if s == 'data':
                prefix = "Data/"  # Data prefix
            else:
                prefix = "MC/mc_" + str(infofile.infos[val]["DSID"]) + "."
            file_string = tuple_path + prefix + val + ".4lep.root"  # file name to open
            count = count_file(file_string, val)  # call the function count_file
            dict_counts[val] = count
        counts[s] = dict_counts

    return counts  # return dictionary of counts


def split_dictionary(original_dict, n):
    """
    Splits a dictionary into multiple segments based on the number of workers specified.

    Args:
        original_dict (dict): Original dictionary containing counts.
        n (int): Number of workers.

    Returns:
        tuple: Tuple containing start and end dictionaries for each segment.
    """
    output_dicts = [{} for _ in range(n + 1)]
    zeros, ones = {}, {}

    for category, sub_dict in original_dict.items():
        for key, value in sub_dict.items():

            # Splitting the value into n parts
            split_values = [value // n] * n
            remaining = value % n

            # Distributing the remainder among the first 'remaining' dictionaries
            for i in range(remaining):
                split_values[i] += 1

            # Set initial dictionary of start points and create dictionary of 1s
            zeros.setdefault(category, {})[key] = 0
            ones.setdefault(category, {})[key] = 1

            output_dicts[0] = zeros

            # Assigning split values to output dictionaries
            for i in range(1, n + 1):
                output_dicts[i].setdefault(category, {})[key] = split_values[i - 1]

    # Make dictionaries cumulative
    for i in range(1, n + 1):
        output_dicts[i] = add_dictionaries(output_dicts[i - 1], output_dicts[i])

    # Create start and end value dictionaries
    start_dicts = output_dicts[:-1]
    end_dicts = output_dicts[1:]

    # Make start values 1 more than previous end values
    for i in range(1, n):
        start_dicts[i] = add_dictionaries(start_dicts[i], ones)

    # Final validation check to ensure all dictionaries add up to original
    if output_dicts[n] == original_dict:
        return start_dicts, end_dicts
    else:
        raise ValueError('End verification failed')


def add_dictionaries(dict1, dict2):
    """
    Add nested dictionaries together.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        dict: Resultant dictionary after addition.
    """
    result = {}
    for key in dict1.keys():
        if key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                # Recursively call for nested dictionaries
                result[key] = add_dictionaries(dict1[key], dict2[key])
            else:
                # Add values if they are not nested dictionaries
                result[key] = dict1[key] + dict2[key]

    return result


if __name__ == '__main__':
    # Define the number of workers
    if float(args.number_workers).is_integer():
        num_workers = int(args.number_workers)
    else:
        raise ValueError('Number of workers must be an integer')

    # Process data and get counts
    counts = get_data_from_files()

    # Split counts into segments based on the number of workers
    start_dicts, end_dicts = split_dictionary(counts, num_workers)

    # Save start and end dictionaries to pickle files
    output_directory = '/home/Nischayee/hzz_analysis/worker_hzz'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, (start_dict, end_dict) in enumerate(zip(start_dicts, end_dicts), 1):
        output_file_path = os.path.join(output_directory, f'data_{i}.pkl')
        with open(output_file_path, 'wb') as output_file:
            pickle.dump((start_dict, end_dict), output_file)

    logging.info('Distributed task data saved.')
