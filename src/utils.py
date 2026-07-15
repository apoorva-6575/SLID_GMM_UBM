"""
Utilities Module.
Provides helper functions for dataset loading, metadata parsing,
label mapping, and configuration parsing.
"""

import csv
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path="d:\\SLID_GMM_UBM\\configs\\config.yaml"):
    """
    Load the YAML configuration file.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config


def get_language_mapping(config):
    """
    Returns dictionaries to map language names to integer labels and vice-versa.
    
    Parameters
    ----------
    config : dict
    
    Returns
    -------
    name_to_label : dict
    label_to_name : dict
    """
    languages = config.get("experiment", {}).get("languages", [])
    if not languages:
        raise ValueError("No languages defined in config.yaml under 'experiment.languages'.")
        
    name_to_label = {lang: i for i, lang in enumerate(languages)}
    label_to_name = {i: lang for i, lang in enumerate(languages)}
    
    return name_to_label, label_to_name


def parse_metadata(split, data_dir="d:\\SLID_GMM_UBM\\data"):
    """
    Parse the metadata.csv for a specific split (train, valid, test).
    
    Parameters
    ----------
    split : str
        'train', 'valid', or 'test'
    data_dir : str
        Base directory of the data folder.
        
    Returns
    -------
    list of dict
        List of rows from the CSV, each containing 'filepath', 'language', 'speaker_id'.
    """
    csv_path = Path(data_dir) / split / "metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")
        
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    return data
