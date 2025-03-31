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
import numpy as np
import argparse
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
import pandas as pd
import argparse

NUM_PROCESS_PER_GPUS = [9, 9, 9, 9]

NUM_TRAINING_EPOCHS = 4000
BATCH_SIZE = 2520
OUTPUT_REPRESENTATION = "natural"
SYNTHETIC_DATASET_FEATURES = [
    {
        "type": "gaussian",
        "scale_min": 0.2,
        "scale_max": 1.0,
        "amplitude_min": 0.5,
        "amplitude_max": 1.5
    }
]
WAVEFORM_TIMESTEPS = 7
USE_ZERO_PADDING = True
USE_CIRCULANT_TRANSLATIONS = False
BASE_FOLDER = "sensitivity_analysis_circulant"

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
        
def load_experiment_specs(exp_name, base_folder=BASE_FOLDER):
    # Create a folder for the experiment
    exp_folder = os.path.join(base_folder, exp_name)

    # Save the experiment specifications to a JSON file
    exp_file_path = os.path.join(exp_folder, "specs.json")
    with open(exp_file_path, "r") as f:
        # Convert any numpy arrays to lists before saving
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.float64):
                return float(obj)
            return obj
        
        specs = json.load(f)
        return specs
        
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

def matrix_cosine_similarity(A, B):
    return np.sum(A * B, axis=(0, 1)) / (np.linalg.norm(A) * np.linalg.norm(B))

def get_model_path(exp_name, epoch):
    return join(BASE_FOLDER, exp_name, "epochs", "ep{}.h5".format(epoch))

def parse_generator_matrix(model, exp_name, epoch):
    model.load_weights(get_model_path(exp_name=exp_name, epoch=epoch))
    lm = model._group_convolution_layer._generator
    lm = lm.numpy()
    return lm

def parse_group_convolution_matrix(model, exp_name, epoch):
    model.load_weights(get_model_path(exp_name=exp_name, epoch=epoch))
    lm = model._group_convolution_layer._lifting_map
    lm = lm.numpy()
    return lm

def find_ideal_translation_generator(learned_generator, circulant=False):
    n = np.shape(learned_generator)[0]
    
    row_idxs = np.arange(0, n)[np.newaxis, :]
    col_idxs = np.arange(0, n)[:, np.newaxis]
    
    if circulant: 
        left_shift = np.where(np.mod(((row_idxs+1) - col_idxs), n) == 0, 1, 0)
        right_shift = np.where(np.mod((row_idxs - (col_idxs +1)), n) == 0, 1, 0)
    else:
        left_shift = np.where(row_idxs+1 == col_idxs, 1, 0)
        right_shift = np.where(row_idxs == col_idxs +1, 1, 0)
    
    sl = matrix_cosine_similarity(left_shift, learned_generator)
    sr = matrix_cosine_similarity(right_shift, learned_generator)
    
    if sl > sr:
        return left_shift
    else:
        return right_shift
    
def form_sensitivity_experiments(estimator_lr_bounds = [1.875e-3, 3.125e-3],
                                      model_lr_bounds = [0.75e-4, 1.25e-4],
                                      lr_decay_bounds = [0.1, 0.2],
                                      alignment_bounds = [0.75, 1.25],
                                      uniformity_bounds = [1.5, 2.5],
                                      resolution_bounds = [0.75, 1.25],
                                      infomax_bounds = [0.75, 1.25],
                                      noise_bounds = [0.0, 0.2],
                                      num_grid_levels = 4,
                                      num_trajectories = 4,
                                      eps = 1e-7):
    problem = {
        'num_vars': 8,
        'names': ['estimator_lr', 
                  'model_lr', 
                  'lr_decay',
                  'alignment',
                  'uniformity',
                  'resolution',
                  'infomax',
                  'noise'],
        'bounds': [estimator_lr_bounds,
                   model_lr_bounds,
                   lr_decay_bounds,
                   alignment_bounds,
                   uniformity_bounds,
                   resolution_bounds,
                   infomax_bounds,
                   noise_bounds]
    }

    # Generate samples using the Morris sampling method.
    hyperparameter_values = morris_sample.sample(problem, 
                                                 N=num_trajectories, 
                                                 num_levels=num_grid_levels)

    
    # For experiment files.    
    experiments = {}
    for exp_idx, vals in enumerate(hyperparameter_values):
        estimator_lr = vals[0]
        model_lr = vals[1]
        lr_decay = vals[2]
        alignment = vals[3]
        uniformity = vals[4]
        resolution = vals[5]
        infomax = vals[6]
        noise = vals[7]
        
        exp_name = f"exp{exp_idx}"
        
        exp_specs = {}
        exp_specs["model_params"] = {
            "zero_padding_size":WAVEFORM_TIMESTEPS,
            "use_zero_padding":USE_ZERO_PADDING,
            "num_uniformity_scales":1,
            "conditional_probability_estimator_hidden_layer_size":WAVEFORM_TIMESTEPS * 16
        }
        exp_specs["data_generator_params"] = {
            "batch_size":BATCH_SIZE,
            "use_circulant_translations":USE_CIRCULANT_TRANSLATIONS,
            "output_representation":"natural",
            "features":SYNTHETIC_DATASET_FEATURES,
            "noise_normalized_std":noise,
            "waveform_timesteps":WAVEFORM_TIMESTEPS
        }
        exp_specs["model_loss_coeffs"] = {
            "alignment_maximization_reg_coeff": alignment,
            "uniformity_maximization_reg_coeff": uniformity,
            "marginal_entropy_minimization_reg_coeff": resolution,
            "joint_entropy_maximization_reg_coeff": (resolution + infomax)
        }
        exp_specs["estimator_loss_coeffs"] = {
            "probability_estimator_entropy_minimization_reg_coeff": 1.0,
            "conditional_probability_estimator_entropy_minimization_reg_coeff": 1.0
        }
        exp_specs["num_training_batches"] = 100
        exp_specs["training_duration_in_epochs"] = NUM_TRAINING_EPOCHS
        exp_specs["model_optimizer_starting_lr"] = model_lr
        exp_specs["model_optimizer_ending_lr"] = model_lr * lr_decay
        exp_specs["estimators_optimizer_starting_lr"] = estimator_lr
        exp_specs["estimators_optimizer_ending_lr"] = estimator_lr * lr_decay
        
        experiments[exp_name] = exp_specs
    
    return experiments

def demean_and_std_normalize(vals, axis=0):
    vals = vals - np.mean(vals, axis=axis)
    vals = vals / np.std(vals, axis=axis)
    return vals

def analyze_experiments(estimator_lr_bounds = [1.875e-3, 3.125e-3],
                        model_lr_bounds = [0.75e-4, 1.25e-4],
                        lr_decay_bounds = [0.1, 0.2],
                        alignment_bounds = [0.75, 1.25],
                        uniformity_bounds = [1.5, 2.5],
                        resolution_bounds = [0.75, 1.25],
                        infomax_bounds = [0.75, 1.25],
                        noise_bounds = [0.0, 0.1],
                        num_grid_levels = 4,
                        num_trajectories = 4,
                        eps = 1e-7):
    
    problem = {
        'num_vars': 8,
        'names': ['estimator_lr', 
                  'model_lr', 
                  'lr_decay',
                  'alignment',
                  'uniformity',
                  'resolution',
                  'infomax',
                  'noise'],
        'bounds': [estimator_lr_bounds,
                   model_lr_bounds,
                   lr_decay_bounds,
                   alignment_bounds,
                   uniformity_bounds,
                   resolution_bounds,
                   infomax_bounds,
                   noise_bounds]
    }

    # Generate samples using the Morris sampling method.
    hyperparameter_values = morris_sample.sample(problem, 
                                                 N=num_trajectories, 
                                                 num_levels=num_grid_levels,
                                                 seed=0)

    # For experiment files.    
    scores = []
    vals = []
    
    for exp_idx, __ in enumerate(hyperparameter_values):
        exp_name = f"exp{exp_idx}"
        exp_specs = load_experiment_specs(exp_name)
        
        assert exp_specs["model_params"]["zero_padding_size"] == WAVEFORM_TIMESTEPS
        assert exp_specs["model_params"]["use_zero_padding"] == USE_ZERO_PADDING
        assert exp_specs["model_params"]["num_uniformity_scales"] == 1
        assert exp_specs["model_params"]["conditional_probability_estimator_hidden_layer_size"] == WAVEFORM_TIMESTEPS * 16
        assert exp_specs["data_generator_params"]["batch_size"] == BATCH_SIZE
        assert exp_specs["data_generator_params"]["use_circulant_translations"] == USE_CIRCULANT_TRANSLATIONS
        assert exp_specs["data_generator_params"]["output_representation"] == "natural"
        assert exp_specs["data_generator_params"]["waveform_timesteps"] == WAVEFORM_TIMESTEPS
        assert exp_specs["estimator_loss_coeffs"]["probability_estimator_entropy_minimization_reg_coeff"] == 1.
        assert exp_specs["estimator_loss_coeffs"]["conditional_probability_estimator_entropy_minimization_reg_coeff"] == 1.
        assert exp_specs["num_training_batches"] == 100
        assert exp_specs["training_duration_in_epochs"] == NUM_TRAINING_EPOCHS
        
        noise = exp_specs["data_generator_params"]["noise_normalized_std"]
        alignment = exp_specs["model_loss_coeffs"]["alignment_maximization_reg_coeff"]
        uniformity = exp_specs["model_loss_coeffs"]["uniformity_maximization_reg_coeff"]
        resolution = exp_specs["model_loss_coeffs"]["marginal_entropy_minimization_reg_coeff"]
        infomax = exp_specs["model_loss_coeffs"]["joint_entropy_maximization_reg_coeff"] - resolution
        model_lr = exp_specs["model_optimizer_starting_lr"]
        lr_decay = exp_specs["model_optimizer_ending_lr"] / model_lr
        estimator_lr = exp_specs["estimators_optimizer_starting_lr"]
        
        assert abs(exp_specs["estimators_optimizer_ending_lr"] - estimator_lr * lr_decay) < eps
        
        vals.append([])
        vals[exp_idx].append(estimator_lr) 
        vals[exp_idx].append(model_lr)
        vals[exp_idx].append(lr_decay)
        vals[exp_idx].append(alignment)
        vals[exp_idx].append(uniformity)
        vals[exp_idx].append(resolution)
        vals[exp_idx].append(infomax)
        vals[exp_idx].append(noise)
        
        model = create_model(**exp_specs["model_params"])
        x_init = np.random.normal(size=(BATCH_SIZE, WAVEFORM_TIMESTEPS, 1))
        model.compile()
        model(x_init)
        
        generator = parse_generator_matrix(model, exp_name, NUM_TRAINING_EPOCHS-1)
        generator = generator[WAVEFORM_TIMESTEPS: -WAVEFORM_TIMESTEPS, WAVEFORM_TIMESTEPS:-WAVEFORM_TIMESTEPS]
        ideal_generator = find_ideal_translation_generator(generator, False)
        cosine_similarity = matrix_cosine_similarity(generator, ideal_generator)
        
        scores.append(cosine_similarity)

    # Normalize values
    vals = np.array(vals)
    scores = np.array(scores)
     
    result = morris_analyze.analyze(problem, vals, scores)

    exps = {}
    exps["Estimator lr"] = vals[:, 0]
    exps["Model lr"] = vals[:, 1]
    exps["Total lr decay"] = vals[:, 2]
    exps["Alignment"] = vals[:, 3]
    exps["Uniformity"] = vals[:, 4]
    exps["Resolution"] = vals[:, 5]
    exps["InfoMax"] = vals[:, 6]
    exps["Noise"] = vals[:, 7]
    exps["Cosine similarity"] = np.array(scores)
    
    pd.DataFrame(exps).to_csv("sensitivity_experiments.csv", float_format='%.3f')
    
    return result

if __name__ == "__main__":
     # Create the parser
    parser = argparse.ArgumentParser(
        description="Run a sensitivity experiment using morris method or analyze previously conducted experiments."
    )

    # Add arguments
    parser.add_argument("--operation", type=str, default='e',
                        help="Operation type. Type 'e' for experiment, 'a' for analysis.")
    parser.add_argument("--num_training_epochs", type=int, default=NUM_TRAINING_EPOCHS,
                        help="Number of training epochs for each experiment.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--output_representation", type=str, default=OUTPUT_REPRESENTATION,
                        help="Output representation type.")
    parser.add_argument("--waveform_timesteps", type=int, default=WAVEFORM_TIMESTEPS,
                        help="Number of timesteps.")
    parser.add_argument("--use_circulant_translations", type=bool, default=USE_CIRCULANT_TRANSLATIONS,
                        help="Use circulant data generation if true.")
    parser.add_argument("--base_folder", type=str, default=BASE_FOLDER,
                        help="Folder in which the experiments will be stored to.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    NUM_TRAINING_EPOCHS = args.num_training_epochs
    BATCH_SIZE = args.batch_size
    OUTPUT_REPRESENTATION = args.output_representation
    WAVEFORM_TIMESTEPS = args.waveform_timesteps
    USE_CIRCULANT_TRANSLATIONS = args.use_circulant_translations
    BASE_FOLDER = args.base_folder
    
    if args.operation == 'e':    
        # For experiments dictionary.
        experiments = form_sensitivity_experiments()

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
    elif args.operation == 'a':
        print(analyze_experiments())
    else:
        raise ValueError("Invalid operation type.")
    
