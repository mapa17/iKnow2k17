# %load solution.py
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


def get_distance(pos0, pos1, method='atom_pos'):
    # Calculate a distance value between e0 and e1 based on
    # method='atom_pos' ... their cummulative difference in atom positions
    # method='mesh_size' ... the abs. diff in mean atom gap size (i.e mesh size)
    # method='mesh_size_variance' ... the abs. diff of variance of the mean atom gap size (i.e variance of the mesh size)
    
    if method == 'atom_pos':
        # Calculate the distance matrix
        D = cdist(pos0, pos1, metric='euclidean')

        # Find the closest match for each point
        assignment = np.argsort(D, axis=1)[:, 0]

        # Calculate distance between each point to its assigned point
        distance = np.sum(np.sqrt(np.sum((pos0 - pos1[assignment, :])**2, axis=1)))
        
    elif method == 'mesh_size':
        # For each atom calculate the mean distance to its three closest neighbours
        D0 = cdist(pos0, pos0, metric='euclidean')
        D0.sort(axis=1)
        D0_mesh_size = np.mean(D0[:, 1:4])

        D1 = cdist(pos1, pos1, metric='euclidean')
        D1.sort(axis=1)
        D1_mesh_size = np.mean(D1[:, 1:4])
        
        distance = np.abs(D0_mesh_size - D1_mesh_size)

    elif method == 'mesh_size_variance':
        # For each atom calculate the mean distance to its three closest neighbours
        D0 = cdist(pos0, pos0, metric='euclidean')
        D0.sort(axis=1)
        D0_mesh_size_var = np.var(np.mean(D0[:, 1:4], axis=1))

        D1 = cdist(pos1, pos1, metric='euclidean')
        D1.sort(axis=1)
        D1_mesh_size_var = np.var(np.mean(D1[:, 1:4], axis=1))
        
        distance = np.abs(D0_mesh_size_var - D1_mesh_size_var)
       
    return distance


def calculate_ranking(prediction_data, lookup_data, distance_method = ''):
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
            d = get_distance(e1pos, e0pos, method=distance_method)
            ranking.append((d, e1))

        ranking.sort()
        results[pre] = ranking
    
    return results


def get_predictions(results, lookup_data):
    # Based on the ranking calculate a energy value for each entry by
    # taking the mean energy value of its 3 closest matches.

    entries = []
    predictions = []
    for entry_id in results.keys():
        entries.append(entry_id)
        closest_entries = [res[1] for res in results[entry_id][0:3]]
        predictions.append(np.mean(get_energies(lookup_data, closest_entries)))
    
    return entries, predictions


def single_stage_prediction(training, validation):
    ranking = calculate_ranking(validation, training, distance_method='atom_pos')   
    entries, predictions = get_predictions(ranking, training)
    return entries, predictions


def two_stage_prediction(training, validation, energy_sw=0.1):
    ranking = calculate_ranking(validation, training, distance_method='atom_pos')   
    entries, predictions = get_predictions(ranking, training)

    # For each entry in the first prediction generate a subset of the training data
    # and apply another distance metric to the subset in order to calculate
    # a improved prediction
    new_predictions = []
    for entry_id, predicted_energy in zip(entries, predictions):
        # Calculate a subset of the data
        training_subset = training[(training['Energy'] > (predicted_energy-energy_sw)) & (training['Energy'] < (predicted_energy+energy_sw))]
        validation_subset = validation[validation['Entry'] == entry_id]

        new_ranking = calculate_ranking(validation_subset, training_subset, distance_method='mesh_size_variance')   
        _, new_prediction = get_predictions(new_ranking, training_subset)
    
        new_predictions.append(new_prediction[0])
    
    return entries, new_predictions


############### HELPER FUNCTIONS - NOT PART OF THE ALGORITHM ###############

def evaluate_prediction(entry_ids, predicted_energies, lookup_table):
    # Calculate the prediction error
    prediction_errors = []
    for entry_id, predicted_energy in zip(entry_ids, predicted_energies):
        real_energy = lookup_table[lookup_table['Entry'] == entry_id]['Energy'].values[0]
        prediction_errors.append(predicted_energy - real_energy)
        
    return np.array(prediction_errors)


def cross_validation(n_tests, n_entries, training_data, prediction_function, kwargs={}):
    prediction_errors = np.zeros(shape=(n_tests, n_entries))    
    for n in range(0, n_tests):
        # Split the training data into a new set of training and validation data in order to test the algorithm
        validation_entries = set(np.random.choice(training_data['Entry'].unique(), n_entries, replace=False))
        training_entries = set(training_data['Entry'].unique()) - validation_entries
        
        print('Running Test (%d/%d) with validation entries %s ...' % (n+1, n_tests, validation_entries))
        
        training = training_data[training_data['Entry'].isin(training_entries)]
        validation = training_data[training_data['Entry'].isin(validation_entries)]

        entries, predictions = prediction_function(training, validation, **kwargs)
        
        prediction_errors[n, :] = evaluate_prediction(entries, predictions, training_data)
    
    return prediction_errors


def get_energies(table, entries):
    return [table[table['Entry'] == entry]['Energy'].values[0] for entry in entries]
        
def get_closest_entries(table, energy):
    uT = table[['Entry', 'Energy']].drop_duplicates()    
    energies = uT['Energy'].values
    entries = uT['Entry'].values    
        
    diff_energies = (energies - energy)**2
    closest_energies = np.argsort(diff_energies)
    closest_entries = entries[closest_energies]
    
    return closest_entries, energies[closest_energies]

