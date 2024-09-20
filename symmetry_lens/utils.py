import numpy as np
from os.path import join
import tensorflow as tf
from symmetry_lens.models import EquivariantAdapter
from symmetry_lens.synthetic_data_generator import SyntheticDataGenerator
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


def create_model(**kwargs):
    model = EquivariantAdapter(**kwargs)
    return model


def make_dataset(
    num_batches=100,
    batch_size=16500,
    features=[
        {"type": "gaussian", "sigmas": [1.0]},
        {"type": "legendre", "l": 2, "m": 1, "lengths": [6, 8]},
    ],
    waveform_timesteps=32,
    noise_normalized_std=0.25,
    use_circulant_translations=False,
    output_representation="natural",
    p_exist = 0.5,
    num_of_lots = 5
):
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

    batches = []
    for b in range(num_batches):
        batches.append(dg.sample_batch_of_data())

    dataset = np.concatenate(batches, axis=0)
    return dataset


def train(
    model,
    dataset,
    batch_size,
    saving_dir=".",
    epochs=1000,
    lr_scheduling = {
        "model_optimizer_starting_lr":1e-4,
        "model_optimizer_ending_lr":1e-5,
        "estimator_optimizer_starting_lr":1e-3,
        "estimator_optimizer_ending_lr":1e-4
        },
    model_loss_coeffs={
        "alignment_maximization_reg_coeff": 1.5,
        "uniformity_maximization_reg_coeff": 2.5,
        "marginal_entropy_minimization_reg_coeff": 1.5,
        "joint_entropy_maximization_reg_coeff": 2.5},
    estimator_loss_coeffs={
        "probability_estimator_entropy_minimization_reg_coeff": 1.0,
        "conditional_probability_estimator_entropy_minimization_reg_coeff": 1.0
    }
):
    # Initialize model.
    model.compile()
    model(dataset[0:batch_size])
    model.summary()
    
    fit(
        model=model,
        optimizer_model=create_optimizer(lr_scheduling["model_optimizer_starting_lr"]),
        optimizer_estimators=create_optimizer(lr_scheduling["estimator_optimizer_starting_lr"]),
        lr_scheduler_model=create_lr_scheduler(
            lr_scheduling["model_optimizer_starting_lr"], lr_scheduling["model_optimizer_ending_lr"], epochs
        ),
        lr_scheduler_estimators=create_lr_scheduler(
            lr_scheduling["estimator_optimizer_starting_lr"], lr_scheduling["estimator_optimizer_ending_lr"], epochs
        ),
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[ModelSavingCallbacks(model, saving_dir)],
        model_loss_coeffs=model_loss_coeffs,
        estimator_loss_coeffs=estimator_loss_coeffs,
    )