import re

import matplotlib.pyplot as plt

# from interpro_scraping import interpro_scraping_pandas
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
import scipy.stats
from datetime import datetime

def clean_up_data_biopy(raw_data, proteins_ids):
    raw_data = raw_data.fillna(0)  # fill nans

    # remove proteins removed in mass spec clean up
    tempdf = proteins_ids.merge(raw_data, how='inner', left_on="Entry", right_on="Entry")
    cleaned_data = tempdf[['Entry', 'Sequence', 'Length', 'Mass']]
    cleaned_data = cleaned_data.fillna(0)

    # turns sequence column into series that will be used to iterate over when calculating biopython features
    sequences = cleaned_data['Sequence']
    # creates dataframe to store biopython features
    seq_data = pd.DataFrame([])
    first_pass = True

    for seq in sequences:
        # determines if X (any aa) or U (seloncysteine) are present in sequence and replaces them with L (leucine most common) or C (cysteine)
        seq = seq.replace("X","L")
        seq = seq.replace("U","C")

        # turns sequence into biopython sequence class
        analyzed_seq = ProteinAnalysis(seq)
        # counts number of each amino acids in sequence returns dic
        aaCount = analyzed_seq.count_amino_acids()
        # calculates percentage of each amino acid in seq, returns dic
        aa_percent = analyzed_seq.get_amino_acids_percent()
        # regularize_aa function is line 16 of this script, reformats biopyhton result by adding ammino acids that aren't present in sequence and fill count with zero
        reg_aa = regularize_aa(aa_percent)
        # alphabetize_aa function is line 26 of this script, orders dictionary in alphabetical order
        aa_values = alphabetize_aa(reg_aa)

        # calculates molecular by replacing X with L if X is present
        mw = analyzed_seq.molecular_weight()  # MW

        # calculates aromaticity of protein
        aromat = analyzed_seq.aromaticity()

        # calculates instability, flex, and gravy while dealing with "X" and "U" by replacing with L and C if present
        instab = analyzed_seq.instability_index()  # float
        flex = analyzed_seq.flexibility()  # returns a list
        gravy = analyzed_seq.gravy()
        # calculates isoelectric point
        iso = analyzed_seq.isoelectric_point()
        # calculates secondary structure presence
        secStruct = analyzed_seq.secondary_structure_fraction()  # tuple of three floats (helix, turn, sheet)
        secStruct_disorder = 1 - sum(secStruct)

        # calculates stats for flex feature
        flex_stat = (np.mean(flex), np.std(flex), np.var(flex), np.max(flex), np.min(flex), np.median(flex))

        # stores all info in dataframe
        temp_df = pd.DataFrame(
            [[seq, *aa_values, mw, aromat, instab, *flex_stat, iso, *secStruct, secStruct_disorder, gravy]])

        # stores all info of every run into collective dataframe
        if first_pass:
            seq_data = temp_df
            first_pass = False
        else:
            seq_data = pd.concat([seq_data, temp_df], axis=0)

    # creates list of column names for dataframe
    aa_list = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    aa_list.sort()
    names_aa = ['frac_aa_' + i for i in aa_list]
    col_names = ['Sequence', *names_aa, 'molecular_weight', 'aromaticity', 'instability_index', 'flexibility_mean',
                 'flexibility_std', 'flexibility_var',
                 'flexibility_max', 'flexibility_min', 'flexibility_median', 'isoelectric_point',
                 'secondary_structure_fraction_helix',
                 'secondary_structure_fraction_turn', 'secondary_structure_fraction_sheet',
                 'secondary_structure_fraction_disordered', 'gravy']
    # changes column names
    seq_data.columns = col_names
    seq_data = seq_data.reset_index(drop=True)

    # adds new features to protein details dataframe
    cleaned_data = pd.merge(cleaned_data, seq_data, on='Sequence')
    cleaned_data = cleaned_data.fillna(0)
    return cleaned_data

def string_to_list(str):
    return [char for char in str]


def find_all(str, sub):
    start = 0
    while True:
        start = str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


def replace_all(list, loc_list, value):
    for index in loc_list:
        list[index] = value
    return list

def regularize_aa(aa_dict):
    aa_list = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    keys = aa_dict.keys()
    if len(keys) != 20:
        for aa in aa_list:
            if aa not in keys:
                aa_dict[aa] = 0
    return aa_dict


def alphabetize_aa(aa_dict):
    aa_list = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    aa_list.sort()
    return [aa_dict[aa] for aa in aa_list]

def clean_up_data_mass_spec(raw_data):
    # calculate average relative abundance from triplicate results
    raw_data["Avg NP Relative Abundance"] = raw_data["NP"]
    # remove any protein with avg zero abundance
    raw_data["Avg NP Relative Abundance"] = raw_data["Avg NP Relative Abundance"].replace(0, np.nan)
    raw_data.dropna(subset=["Avg NP Relative Abundance"], inplace=True)
    # calculate percent relative abundance
    Abudance_sum = raw_data["Avg NP Relative Abundance"].sum()
    raw_data['NP_%_Abundance'] = (raw_data["Avg NP Relative Abundance"] / Abudance_sum) * 100
    # calculate enrichement
    raw_data["Enrichment"] = np.log2(raw_data["Avg NP Relative Abundance"] / raw_data["Serum"])

    return raw_data[["Accession", "NP_%_Abundance", "Enrichment", "Serum"]]

def normalize_mass_length_1DF(df1):
    max_length = df1['Length'].max()
    max_mass = df1['Mass'].max()
    max_mw = df1['molecular_weight'].max()

    df1['length'] = df1['Length'] / max_length
    df1['mass'] = df1['Mass'] / max_mass
    df1['molecular_weight'] = df1['molecular_weight'] / max_mw

    return df1

if __name__ == "__main__":
    print(type(ProteinAnalysis))
    print('found biopython')