EXPERIMENTS_FOLDER = "/home/oefe/Code/SymmetryLens/ablation_study3"

EXP_NAME = "ablated:[]"
EPOCH = 2900
import numpy as np
import sys
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from symmetry_lens import *

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
import tensorflow as tf
import shutil
from scipy.linalg import logm
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import differential_entropy
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
import json
from matplotlib.ticker import MaxNLocator, MultipleLocator

TRANSLATION_DIRECTION_LEFT_TO_RIGHT = 0
TRANSLATION_DIRECTION_RIGHT_TO_LEFT = 1
RESOLUTION_FILTER_POLARITY_NEGATIVE = 0
RESOLUTION_FILTER_POLARITY_POSITIVE = 1

TRANSLATION_DIRECTION = TRANSLATION_DIRECTION_LEFT_TO_RIGHT
RESOLUTION_FILTER_POLARITY = RESOLUTION_FILTER_POLARITY_NEGATIVE

sns.set_theme(
    style="whitegrid",
    rc={
        'text.color': 'black',         # Set overall text color
        'axes.labelsize': 39,             # Font size for x and y labels
        'axes.labelcolor': 'black',    # Color for x and y labels
        'axes.titlesize': 48,             # Font size for the plot title
        'axes.titlecolor': 'black',    # Color for the plot title
        'legend.fontsize': 39,            # Font size for legend texts
        'legend.edgecolor': 'black',   # (Optional) Legend edge color
        'xtick.labelsize': 30,            # Font size for x-axis tick labels
        'xtick.color': 'black',        # Color for x-axis tick labels
        'ytick.labelsize': 30,            # Font size for y-axis tick labels
        'ytick.color': 'black',        # Color for y-axis tick labels
    }
)

def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file: {file_path}")
        return None
    
def _load_experiment_specs():
    specs_path = join(EXPERIMENTS_FOLDER, EXP_NAME, "specs.json")
    return read_json(specs_path)
    

def _get_model_dir(epoch=EPOCH):
    return join(EXPERIMENTS_FOLDER, EXP_NAME, "epochs", "ep{}.h5".format(epoch))

def _get_snapshot_dir(epoch=EPOCH):
    return join(EXPERIMENTS_FOLDER, EXP_NAME, "tensor_snapshots", "ep{}".format(epoch))

def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file: {file_path}")
        return None
    
def _load_experiment_specs():
    specs_path = join(EXPERIMENTS_FOLDER, EXP_NAME, "specs.json")
    return read_json(specs_path)

def _make_output_dirs(epoch=EPOCH):
    if exists(_get_snapshot_dir(epoch=epoch)):
        shutil.rmtree(_get_snapshot_dir(epoch=epoch))

    makedirs(_get_snapshot_dir(epoch=epoch), exist_ok=True)

def _plot_matrix_comparison(ideal_matrix, learned_matrix, save_path):
    n_timesteps = np.shape(ideal_matrix)[0]
        
    component_labels = np.arange(start=0, stop=n_timesteps)
    interval = 1 + n_timesteps // 4
    xtick_labels = np.where(component_labels % interval == 0, component_labels.astype(str), "")
    ytick_labels = np.where(component_labels % interval == 0, component_labels.astype(str), "")
    
    # Calculate the error matrix (difference between ideal and learned matrices)
    error_matrix = ideal_matrix - learned_matrix
    error_values = error_matrix.flatten()

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Determine symmetric color limits based on the largest absolute value in ideal_matrix
    mag = np.max(np.abs(ideal_matrix))

    # ---- Ideal matrix ----
    sns.heatmap(
        ideal_matrix,
        ax=axes[0, 0],
        cmap=sns.color_palette("RdBu_r", as_cmap=True),
        vmin=-mag,
        vmax=mag,
        xticklabels=xtick_labels,
        yticklabels=ytick_labels,
        # Force symmetric ticks and consistent formatting
        cbar=True,
        cbar_kws={
            "ticks": np.linspace(-mag, mag, 5),      # e.g., 5 ticks from -mag to +mag
            "format": StrMethodFormatter("{x:+.2f}") # show +/- sign with 2 decimals
        }
    )
    axes[0, 0].set_title("Ideal", pad=12)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("")

    # ---- Learned matrix ----
    sns.heatmap(
        learned_matrix,
        ax=axes[0, 1],
        cmap=sns.color_palette("RdBu_r", as_cmap=True),
        vmin=-mag,
        vmax=mag,
        xticklabels=xtick_labels,
        yticklabels=ytick_labels,
        cbar=True,
        cbar_kws={
            "ticks": np.linspace(-mag, mag, 5),
            "format": StrMethodFormatter("{x:+.2f}")
        }
    )
    axes[0, 1].set_title("Learned", pad=12)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")
    
    # ---- Error matrix ----
    # Because your error_matrix is already constrained to [-0.5, 0.5], specify those ticks:
    sns.heatmap(
        error_matrix,
        ax=axes[1, 0],
        cmap=sns.color_palette("RdBu_r", as_cmap=True),
        vmin=-1.01,
        vmax=1.01,
        xticklabels=xtick_labels,
        yticklabels=ytick_labels,
        cbar=True,
        cbar_kws={
            "ticks": np.linspace(-1.0, 1.0, 7),
            "format": StrMethodFormatter("{x:+.2f}")
        }
    )
    axes[1, 0].set_title("Error", pad=12)
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("")
    
    # ---- Error histogram ----
    sns.histplot(
        error_values, 
        ax=axes[1, 1],
        kde=False,
        binwidth=0.05,
        color=sns.color_palette("Blues")[4]
    )
    axes[1, 1].set_xlim(-1.0, 1.0)
    axes[1, 1].set_title("Error Histogram", pad=12)
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()

def _matrix_cosine_similarity(A, B):
    return np.sum(A * B, axis=(0,1)) / (np.linalg.norm(A) * np.linalg.norm(B))

def _parse_learned_generator(epoch=EPOCH):
    model.load_weights(_get_model_dir(epoch=epoch))
    learned_generator = model._group_convolution_layer._generator
    learned_generator = learned_generator.numpy()
    return learned_generator

def _parse_group_convolution_matrix(epoch=EPOCH):
    lm = model._group_convolution_layer._lifting_map
    lm = lm.numpy()
    return lm

def _find_ideal_circulant_translation_generator(learned_generator, circulant=False):
    n = np.shape(learned_generator)[0]
    
    row_idxs = np.arange(0, n)[np.newaxis, :]
    col_idxs = np.arange(0, n)[:, np.newaxis]
    
    if circulant: 
        left_shift = np.where(((row_idxs+1) - col_idxs) % n == 0, 1, 0)
        right_shift = np.where((row_idxs - (col_idxs +1)) % n == 0, 1, 0)
    else:
        left_shift = np.where(row_idxs+1 == col_idxs, 1, 0)
        right_shift = np.where(row_idxs == col_idxs +1, 1, 0)
    
    sl = _matrix_cosine_similarity(left_shift, learned_generator)
    sr = _matrix_cosine_similarity(right_shift, learned_generator)
    
    if sl > sr:
        return left_shift
    else:
        return right_shift

def _find_ideal_group_convolution_matrix(learned_group_convolution_matrix):
    n = np.shape(learned_generator)[0]
    
    row_idxs = np.arange(0, n)[np.newaxis, :]
    col_idxs = np.arange(0, n)[:, np.newaxis]
    
    reverting = np.where(row_idxs == col_idxs, 1, 0)
    identity = np.where(row_idxs + col_idxs == n, 1, 0)

    sr = _matrix_cosine_similarity(reverting, learned_group_convolution_matrix)
    si = _matrix_cosine_similarity(identity, learned_group_convolution_matrix)
    
    if np.abs(sr) > np.abs(si):
        if sr > 0.:
            return reverting
        else:
            return -reverting
    else:
        if si > 0.:
            return identity
        else:
            return -identity

specs = _load_experiment_specs()

use_zero_padding = specs["model_params"]["use_zero_padding"]
batch_size = specs["data_generator_params"]["batch_size"]
waveform_timesteps = specs["data_generator_params"]["waveform_timesteps"]

# Create model and load weights.
x_init = np.random.normal(size=(specs["data_generator_params"]["batch_size"], 
                                specs["data_generator_params"]["waveform_timesteps"], 
                                1))

model = create_model(zero_padding_size=specs["data_generator_params"]["waveform_timesteps"],
                     use_zero_padding=specs["model_params"]["use_zero_padding"],
                     conditional_probability_estimator_hidden_layer_size=specs["model_params"]["conditional_probability_estimator_hidden_layer_size"],
x_init = np.random.normal(size=(batch_size, waveform_timesteps, 1))
model = create_model(zero_padding_size=waveform_timesteps,
                     use_zero_padding=use_zero_padding,
                     conditional_probability_estimator_hidden_layer_size = waveform_timesteps*16,
                     num_uniformity_scales=1)
model.compile()
model(x_init)

learned_generator = _parse_learned_generator()
if specs["model_params"]["use_zero_padding"]:
    learned_generator = learned_generator[specs["data_generator_params"]["waveform_timesteps"]:-specs["data_generator_params"]["waveform_timesteps"], 
                                          specs["data_generator_params"]["waveform_timesteps"]:-specs["data_generator_params"]["waveform_timesteps"]]

if use_zero_padding:
    learned_generator = learned_generator[waveform_timesteps:-waveform_timesteps, 
                                          waveform_timesteps:-waveform_timesteps]

group_convolution_matrix = _parse_group_convolution_matrix()
    
ideal_generator = _find_ideal_circulant_translation_generator(learned_generator, circulant=False)
ideal_group_convolution_matrix = _find_ideal_group_convolution_matrix(group_convolution_matrix)

cosine_similarity = _matrix_cosine_similarity(learned_generator, ideal_generator)

print(f"Cosine similarity: {_matrix_cosine_similarity(learned_generator, ideal_generator)}")

_make_output_dirs()
_plot_matrix_comparison(ideal_generator, 
                        learned_generator, 
                        join(_get_snapshot_dir(), f"generator_comparison.png"))

_plot_matrix_comparison(ideal_group_convolution_matrix, 
                        group_convolution_matrix, 
                        join(_get_snapshot_dir(), f"group_convolution_matrix_comparison.png"))



