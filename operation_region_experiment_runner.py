import tensorflow as tf
import numpy as np
import multiprocessing
import time
import sys
import os
import json
from symmetry_lens import *
from os import makedirs
from os.path import join, exists
from shutil import rmtree

NUM_PROCESS_PER_GPUS = [1, 1, 1, 1]

NUM_TRAINING_EPOCHS = 10000
BATCH_SIZE = 10800
OUTPUT_REPRESENTATION = "natural"
SYNTHETIC_DATASET_FEATURES = [
    {
        "type": "mnist_slices"
    }
]
NOISE_STD = 0.0
WAVEFORM_TIMESTEPS = 27
USE_ZERO_PADDING = True
CONDITIONAL_PROBABILITY_ESTIMATOR_HIDDEN_LAYER_SIZE = WAVEFORM_TIMESTEPS*16
USE_CIRCULANT_TRANSLATIONS = False
BASE_FOLDER = "operating_region_analysis"

def get_exp_dir(exp_name):
    model_dir = join(BASE_FOLDER, exp_name)
    return model_dir


def get_model_weights_saving_dir(exp_name):
    model_dir = join(BASE_FOLDER, exp_name, "epochs")
    return model_dir


def set_visible_gpu(gpu_index):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No GPUs found.")

    if gpu_index >= len(gpus):
        raise ValueError(f"Invalid GPU index: {gpu_index}. Available GPUs: {len(gpus)}")

    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_index], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        print(f"Using GPU: {gpus[gpu_index]}")
    except RuntimeError as e:
        print(e)


def wait_for_gpu_initialization(gpu_index):
    while True:
        try:
            # Try to create a small tensor on the GPU to force initialization
            with tf.device(f"/GPU:{gpu_index}"):
                tf.constant([1.0, 2.0, 3.0])
            print(f"GPU {gpu_index} initialized successfully.")
            break
        except RuntimeError as e:
            print(f"Waiting for GPU {gpu_index} to initialize...")
            time.sleep(1)


def save_experiment_specs(exp_name, exp_specs, base_folder=BASE_FOLDER):
    # Create a folder for the experiment
    exp_folder = os.path.join(base_folder, exp_name)

    # Save the experiment specifications to a JSON file
    exp_file_path = os.path.join(exp_folder, "specs.json")
    with open(exp_file_path, "w") as f:
        # Convert any numpy arrays to lists before saving
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.float64):
                return float(obj)
            return obj

        json.dump(exp_specs, f, default=convert, indent=4)
        
def train_model(exp_name, exp_specs, gpu_index):
    exp_dir = get_exp_dir(exp_name)
    if exists(exp_dir):
        rmtree(exp_dir)

    makedirs(exp_dir, exist_ok=True)  
    
    log_folder = os.path.join("sensitivity_analysis", exp_name)
    sys.stdout = open(os.path.join(log_folder, f"training_log.txt"), "w", buffering=1)
    sys.stderr = open(
        os.path.join(log_folder, f"training_error_log.txt"), "w", buffering=1
    )

    set_visible_gpu(gpu_index)
    wait_for_gpu_initialization(gpu_index)
    save_experiment_specs(exp_name, exp_specs)

    model = create_model(**exp_specs["model_params"])
    data_generator = make_data_generator(**exp_specs["data_generator_params"])
    
    makedirs(get_model_weights_saving_dir(exp_name), exist_ok=True)

    train(
        model,
        data_generator=data_generator,
        saving_dir=get_model_weights_saving_dir(exp_name),
        num_training_batches=exp_specs["num_training_batches"],
        epochs=exp_specs["training_duration_in_epochs"],
        model_optimizer_starting_lr=exp_specs["model_optimizer_starting_lr"],
        model_optimizer_ending_lr=exp_specs["model_optimizer_ending_lr"],
        estimators_optimizer_starting_lr=exp_specs["estimators_optimizer_starting_lr"],
        estimators_optimizer_ending_lr=exp_specs["estimators_optimizer_ending_lr"],
        model_loss_coeffs=exp_specs["model_loss_coeffs"],
        estimator_loss_coeffs=exp_specs["estimator_loss_coeffs"],
    )

    print(f"Model for experiment {exp_name} trained on GPU {gpu_index}")
    
def find_gpu_index(process_index):
    gpu_index = -1
    for i in range(4):
        if sum(NUM_PROCESS_PER_GPUS[0 : i + 1]) > process_index:
            gpu_index = i
            break

    if gpu_index != -1:
        return gpu_index
    else:
        raise RuntimeError("GPU process capacity have been exceeded.")


def distribute_experiments(experiments):
    tasks = []
    for i, (exp_name, exp_specs) in enumerate(experiments.items()):
        gpu_index = find_gpu_index(i)
        tasks.append((exp_name, exp_specs, gpu_index))

    return tasks

def form_operating_region_experiments(default_estimator_lr = 2.5e-3,
                                      default_model_lr = 1e-4,
                                      default_lr_decay = 0.1,
                                      default_alignment = 1.0,
                                      default_uniformity = 2.0,
                                      default_resolution = 1.0,
                                      default_infomax = 1.0,
                                      default_noise = 0.0,
                                      min_multiplier = 1.,
                                      max_multipler = 1.,
                                      num_vals = 1,
                                      eps = 1e-7):

    multipliers = np.logspace(np.log10(min_multiplier), 
                              np.log10(max_multipler), 
                              num_vals)
    
    default_hyperparameters = {"estimator_lr":default_estimator_lr, 
                               "model_lr":default_model_lr, 
                               "lr_decay":default_lr_decay, 
                               "alignment":default_alignment, 
                               "uniformity":default_uniformity, 
                               "resolution":default_resolution, 
                               "infomax":default_infomax,
                               "noise":default_noise}

    # For experiment files.    
    experiments = {}
    for key in list(default_hyperparameters.keys()):
        # Set default parameter values.
        hyperparams = default_hyperparameters

        for val in multipliers:
            if np.abs(val - 1.) < eps:
                continue
            
            hyperparams[key] = val
            exp_name = f"modified:{key}__value:{val}"
            exp_specs = {}
            exp_specs["model_params"] = {
                "zero_padding_size":WAVEFORM_TIMESTEPS,
                "use_zero_padding":USE_ZERO_PADDING,
                "num_uniformity_scales":1,
                "conditional_probability_estimator_hidden_layer_size":CONDITIONAL_PROBABILITY_ESTIMATOR_HIDDEN_LAYER_SIZE
            }
            exp_specs["data_generator_params"] = {
                "batch_size":BATCH_SIZE,
                "use_circulant_translations":USE_CIRCULANT_TRANSLATIONS,
                "output_representation":"natural",
                "features":SYNTHETIC_DATASET_FEATURES,
                "noise_normalized_std":hyperparams["noise"],
                "waveform_timesteps":WAVEFORM_TIMESTEPS
            }
            exp_specs["model_loss_coeffs"] = {
                "alignment_maximization_reg_coeff": hyperparams["alignment"],
                "uniformity_maximization_reg_coeff": hyperparams["uniformity"],
                "marginal_entropy_minimization_reg_coeff": hyperparams["resolution"],
                "joint_entropy_maximization_reg_coeff": (hyperparams["resolution"] + hyperparams["infomax"])
            }
            exp_specs["estimator_loss_coeffs"] = {
                "probability_estimator_entropy_minimization_reg_coeff": 1.0,
                "conditional_probability_estimator_entropy_minimization_reg_coeff": 1.0
            }
            exp_specs["num_training_batches"] = 100
            exp_specs["training_duration_in_epochs"] = NUM_TRAINING_EPOCHS
            exp_specs["model_optimizer_starting_lr"] = hyperparams["model_lr"]
            exp_specs["model_optimizer_ending_lr"] = hyperparams["model_lr"] * hyperparams["lr_decay"]
            exp_specs["estimators_optimizer_starting_lr"] = hyperparams["estimator_lr"]
            exp_specs["estimators_optimizer_ending_lr"] = hyperparams["estimator_lr"] * hyperparams["lr_decay"]
            
            experiments[exp_name] = exp_specs
    
    return experiments
        

if __name__ == "__main__":
    # For experiments dictionary.
    experiments = form_operating_region_experiments()
    
    # Number of GPUs available
    gpus = tf.config.experimental.list_physical_devices("GPU")
    num_gpus = len(gpus)

    # Initialize GPUs.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    # Distribute experiments to GPUs
    tasks = distribute_experiments(experiments)

    # Use the 'spawn' method to create a new Python process
    multiprocessing.set_start_method("spawn")

    num_processes = sum(NUM_PROCESS_PER_GPUS)
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the training function to each task
        pool.starmap(train_model, tasks)

    print("All experiments have been completed.")