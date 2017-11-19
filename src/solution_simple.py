# Import important libraries
%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from itertools import chain
from itertools import repeat
from collections import OrderedDict

import xml.etree.ElementTree as ET

config = {}

# E.g. "1.256660 0.431805 -4.981400"
def parse_coords(text):
    return [float(x) for x in text.split(' ')]

def iter_dataset(xml_tree):
    for child in xml_tree.getroot():
        name = int(child.tag.split('_')[1])
        try:
            energy =  float(child.find('energy').text)
        except AttributeError:
            energy = np.nan
        atoms = [parse_coords(element.text) for element in child.find('coordinates').findall('c')]
        for i, coords in enumerate(atoms):
            yield {'Entry':name, 'Energy':energy, 'Atom': i, 'X':coords[0], 'Y':coords[1], 'Z':coords[2]}

def parse_dataset(xml_file):
    xml_tree = ET.parse(xml_file)
    training_set = list(iter_dataset(xml_tree))

    return pd.DataFrame(training_set, columns=('Entry', 'Energy', 'Atom', 'X', 'Y', 'Z'))
    
def get_pos(data, entry):
    # Convert the X, Y, Z position for entry to a numpy array of size 60x3
    
    # Get single entry
    E = data[data['Entry'] == entry]
    if E.empty:
        print('Invalid Entry id!')
        return None
    
    # Get the position in format Nx3
    E_ = E.apply(lambda row: [row['X'], row['Y'], row['Z']], axis=1).values
    
    # Transform it to a numpy array
    Epos = np.reshape(list(chain(*E_)), (60, 3))
    
    return Epos


def get_distance(pos0, pos1):
    # Calculate a distance value between e0 and e1 based on their atom positions
    
    # Calculate the distance matrix
    D = cdist(pos0, pos1, metric='euclidean')
    
    # Find the closest match for each point
    assignment = np.argsort(D, axis=1)[:, 0]
    
    # Calculate distance between each point to its assigned point
    cum_distance = np.sum(np.sqrt(np.sum((pos0 - pos1[assignment, :])**2, axis=1)))
    
    # Return cummulative distance between assignt points
    return cum_distance


def calculate_ranking(prediction_data, lookup_data):
    # For each entry in 'prediction_data' rank all entries in 'data'
    #
    # Return a ordered Dictionary containg for each prediction_data Entry
    # a tuple describing the similary/distance to each entry in the lookup table.
    
    prediction_entries = prediction_data['Entry'].drop_duplicates()
    lookup_entries = lookup_data['Entry'].drop_duplicates()
    
    results = OrderedDict()
    for pre in prediction_entries:
        ranking = []
        e0pos = get_pos(prediction_data, pre)
        for (e0, e1) in zip(repeat(pre), lookup_entries):
            e1pos = get_pos(lookup_data, e1)
            d = get_distance(e1pos, e0pos)
            ranking.append((d, e1))

        ranking.sort()
        results[pre] = ranking
    
    return results


def get_predictions(ranking, lookup_data):
    entries = []
    predictions = []
    for entry_id in ranking.keys():
        entries.append(entry_id)
        closest_entries = [res[1] for res in ranking[entry_id][0:3]]
        predictions.append(np.mean(get_energies(lookup_data, closest_entries)))
    
    return entries, predictions

############### HELPER FUNCTIONS - NOT PART OF THE ALGORITHM ###############

def get_energies(table, entries):
    return [table[table['Entry'] == entry]['Energy'].values[0] for entry in entries]

