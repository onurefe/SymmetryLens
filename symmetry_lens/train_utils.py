import numpy as np
from os.path import join
import tensorflow as tf
from symmetry_lens.models import EquivariantAdapter, EquivariantAdapter2d
from symmetry_lens.synthetic_data_generator import SyntheticDataGenerator, SyntheticDataGenerator2d
from symmetry_lens.training_loop import fit


class LogarithmicLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, final_lr, epochs):
        super(LogarithmicLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.epochs = epochs
        self.lrs = self.calculate_lrs()

    def calculate_lrs(self):
        return np.logspace(
            np.log10(self.initial_lr), np.log10(self.final_lr), self.epochs
        )

    def get_learning_rate(self, epoch, logs=None):
        return self.lrs[epoch]


class ModelSavingCallbacks:
    def __init__(self, model, dir):
        self._model = model
        self._dir = dir

    def on_epoch_end(self, epoch, logs=None):
        path = join(self._dir, "ep{}.h5".format(epoch))
        self._model.save_weights(path)


def create_optimizer(lr=1e-4):
    optimizer = tf.optimizers.Adam(lr)
    return optimizer


def create_lr_scheduler(initial_lr, final_lr, epochs):
    lr_scheduler = LogarithmicLearningRateScheduler(initial_lr, final_lr, epochs)
    return lr_scheduler


def create_model(dims=1, model_2d_type=None, **kwargs):
    if dims == 1:
        model = EquivariantAdapter(**kwargs)
    elif dims == 2:
        model = EquivariantAdapter2d(**kwargs)
    else:
        raise Exception("More than 2 generators are not implemented.")
        
    return model


def make_data_generator(
    dims=1,
    batch_size=16000,
    features=[
        {"type": "gaussian", "sigmas": [1.0]},
        {"type": "legendre", "l": 2, "m": 1, "lengths": [6, 8]},
    ],
    waveform_timesteps=32,
    latent_x_dims = 7,
    latent_y_dims = 7,
    flatten_output = True,
    noise_normalized_std=0.25,
    use_circulant_translations=False,
    output_representation="natural",
    p_exist = 0.5,
    num_of_lots = 5
):
    if dims == 1:
        dg = SyntheticDataGenerator(
            batch_size=batch_size,
            waveform_timesteps=waveform_timesteps,
            noise_normalized_std=noise_normalized_std,
            output_representation=output_representation,
            use_circulant_translations=use_circulant_translations,
            features=features,
            p_exist=p_exist,
            num_of_lots=num_of_lots
        )
    elif dims == 2:
        dg = SyntheticDataGenerator2d(
            batch_size=batch_size,
            latent_x_dims=latent_x_dims,
            latent_y_dims=latent_y_dims,
            noise_normalized_std=noise_normalized_std,
            output_representation=output_representation,
            flatten_output=flatten_output,
            features=features,
            p_exist=p_exist,
            num_of_lots=num_of_lots
        )

    return dg

def train(
    model,
    data_generator,
    num_training_batches=100,
    saving_dir=".",
    epochs=1000,
    model_optimizer_starting_lr=1e-4,
    model_optimizer_ending_lr=1e-5,
    estimators_optimizer_starting_lr=1e-3,
    estimators_optimizer_ending_lr=1e-4,
    model_loss_coeffs= {
        "alignment_maximization_reg_coeff": 1.0,
        "uniformity_maximization_reg_coeff": 2.0,
        "marginal_entropy_minimization_reg_coeff": 1.0,
        "joint_entropy_maximization_reg_coeff": 1.75
    },
    estimator_loss_coeffs= {
        "probability_estimator_entropy_minimization_reg_coeff": 1.0,
        "conditional_probability_estimator_entropy_minimization_reg_coeff": 1.0
    }
):
    # Create dataset upfront for speed.
    batches = []
    for b in range(num_training_batches):
        batches.append(data_generator.sample_batch_of_data())

    dataset = np.concatenate(batches, axis=0)

    # Initialize model.
    model.compile()
    model(data_generator.sample_batch_of_data())
    model.summary()

    fit(
        model=model,
        optimizer_model=create_optimizer(model_optimizer_starting_lr),
        optimizer_estimators=create_optimizer(estimators_optimizer_starting_lr),
        lr_scheduler_model=create_lr_scheduler(
            model_optimizer_starting_lr, model_optimizer_ending_lr, epochs
        ),
        lr_scheduler_estimators=create_lr_scheduler(
            estimators_optimizer_starting_lr, estimators_optimizer_ending_lr, epochs
        ),
        dataset=dataset,
        batch_size=data_generator.batch_size,
        epochs=epochs,
        callbacks=[ModelSavingCallbacks(model, saving_dir)],
        model_loss_coeffs=model_loss_coeffs,
        estimator_loss_coeffs=estimator_loss_coeffs,
    )
