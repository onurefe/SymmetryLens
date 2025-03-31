EXPERIMENTS_FOLDER = "/home/oefe/Code/SymmetryLens/baseline"

EXP_NAME = "exp_9dims_sgan"
EPOCH = 80
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
from symmetry_gan import define_generator, define_discriminator, define_gan

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

def _get_model_dir(epoch=EPOCH):
    return join(EXPERIMENTS_FOLDER, EXP_NAME, "epochs", "ep{}.h5".format(epoch))

def _get_snapshot_dir(epoch=EPOCH):
    return join(EXPERIMENTS_FOLDER, EXP_NAME, "tensor_snapshots", "ep{}".format(epoch))

def _make_output_dirs(epoch=EPOCH):
    if exists(_get_snapshot_dir(epoch=epoch)):
        shutil.rmtree(_get_snapshot_dir(epoch=epoch))

    makedirs(_get_snapshot_dir(epoch=epoch), exist_ok=True)

def _plot_generator_matrix_comparison(ideal_matrix, learned_matrix, save_path):
    n_timesteps = np.shape(ideal_matrix)[0]
        
    component_labels = np.arange(start=0, stop=n_timesteps)
    interval = 1 + n_timesteps // 9
    
    component_labels = np.arange(start=0, stop=n_timesteps)
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
    # Because your error_matrix is already constrained to [-0.15, 0.15], specify those ticks:
    sns.heatmap(
        error_matrix,
        ax=axes[1, 0],
        cmap=sns.color_palette("RdBu_r", as_cmap=True),
        vmin=-1.,
        vmax=1.,
        xticklabels=xtick_labels,
        yticklabels=ytick_labels,
        cbar=True,
        cbar_kws={
            "ticks": np.linspace(-1., 1., 7),
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
        color=sns.color_palette("Blues")[4]
    )
    axes[1, 1].set_xlim(-1., 1.)
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

def _parse_learned_generator(epoch):
    gan.load_weights(_get_model_dir(epoch=epoch))
    learned_generator = generator.layers[1].compute_generator()
    learned_generator = learned_generator.numpy()
    return learned_generator

def _find_ideal_circulant_translation_generator(learned_generator):
    n = np.shape(learned_generator)[0]
    
    max_score = -1.
    ideal_generator = None
    
    for s in range(0, n):
        candidate_ideal_generator = np.roll(np.eye(n), axis=1, shift=s)
        score = _matrix_cosine_similarity(learned_generator, candidate_ideal_generator)
        if score > max_score:
            ideal_generator = candidate_ideal_generator
            max_score = score            
        
    return ideal_generator

generator = define_generator()
discriminator = define_discriminator()
gan = define_gan(generator=generator, discriminator=discriminator)

max_cosine_similarity = -1.
best_epoch = None

for epoch in range(0, 100):
    learned_generator = _parse_learned_generator(epoch)
    ideal_generator = _find_ideal_circulant_translation_generator(learned_generator)
    cosine_similarity = _matrix_cosine_similarity(learned_generator, ideal_generator)
    
    if cosine_similarity > max_cosine_similarity:
        max_cosine_similarity = cosine_similarity
        best_epoch = epoch

learned_generator = _parse_learned_generator(best_epoch)
ideal_generator = _find_ideal_circulant_translation_generator(learned_generator)

print(f"Cosine similarity: {_matrix_cosine_similarity(learned_generator, ideal_generator)}")
print(f"Best epoch:{best_epoch}")

_make_output_dirs()
_plot_generator_matrix_comparison(ideal_generator, 
                                  learned_generator, 
                                  join(_get_snapshot_dir(), f"generator_comparison.png"))

