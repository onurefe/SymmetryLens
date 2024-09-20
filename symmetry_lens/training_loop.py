import tensorflow as tf
import numpy as np
from symmetry_lens.regularizations import RegularizationOrders as ro

def get_ordered_regularizations_coeffs(
    alignment_maximization_reg_coeff=0.0,
    uniformity_maximization_reg_coeff=0.0,
    marginal_entropy_minimization_reg_coeff=0.0,
    joint_entropy_maximization_reg_coeff=0.0,
    probability_estimator_entropy_minimization_reg_coeff=0.0,
    conditional_probability_estimator_entropy_minimization_reg_coeff=0.0
):
    reg_coeffs = [
        [ro.ALIGNMENT_MAXIMIZATION_REGULARIZATION_ORDER, alignment_maximization_reg_coeff],
        [
            ro.UNIFORMITY_MAXIMIZATION_REGULARIZATION_ORDER,
            uniformity_maximization_reg_coeff,
        ],
        [
            ro.MARGINAL_ENTROPY_MINIMIZATION_REGULARIZATION_ORDER,
            marginal_entropy_minimization_reg_coeff,
        ],
        [
            ro.JOINT_ENTROPY_MAXIMIZATION_REGULARIZATION_ORDER,
            joint_entropy_maximization_reg_coeff,
        ],
        [
            ro.PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER,
            probability_estimator_entropy_minimization_reg_coeff,
        ],
        [
            ro.CONDITIONAL_PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER,
            conditional_probability_estimator_entropy_minimization_reg_coeff
        ]
    ]

    regularization_coeffs = tf.convert_to_tensor(reg_coeffs, dtype=tf.float32)
    sorting_idxs = tf.argsort(regularization_coeffs[:, 0], axis=0)
    regularization_coeffs = tf.gather(regularization_coeffs[:, 1], sorting_idxs)

    return regularization_coeffs


def get_ordered_regularization_terms(model):
    regularization_tensor = tf.convert_to_tensor(model.losses)
    sorter = tf.argsort(regularization_tensor[:, 0], axis=0)
    orders_sorted = tf.cast(tf.gather(regularization_tensor[:, 0], sorter), tf.int32)
    values_sorted = tf.gather(regularization_tensor[:, 1], sorter)

    merged_dims = tf.reduce_sum(tf.where((orders_sorted[1:] != orders_sorted[:-1]), 1, 0)) + 1
    merged_idxs = tf.range(0, merged_dims, dtype=tf.int32)

    orders_sorted = orders_sorted - tf.reduce_min(orders_sorted)
    merging_mask = tf.where(merged_idxs[tf.newaxis, :] == orders_sorted[:, tf.newaxis], 1., 0.)
    merged_regularizations = tf.matmul(tf.expand_dims(values_sorted, axis=0), merging_mask)
    
    return merged_regularizations[0]


def get_loss(regularizations, **kwargs):
    reg_coeffs = get_ordered_regularizations_coeffs(**kwargs)
    return tf.reduce_sum(regularizations * reg_coeffs)


def split_weights(weight_list, included_modules):
    included_weights = []
    excluded_weights = []
    for weight in weight_list:
        name_list = weight.name.split("/")
        included = False

        for module in included_modules:
            for name in name_list:
                if module in name[0:len(module)]:
                    included = True
                
            if included:
                break

        if included:
            included_weights.append(weight)
        else:
            excluded_weights.append(weight)

    return included_weights, excluded_weights

def compute_total_lr_scaled_training_steps(num_epochs, num_steps, lr_scheduler):
    lr_scaled_training_steps = 0.0

    for epoch in range(num_epochs):
        lr = lr_scheduler.get_learning_rate(epoch)
        lr_scaled_training_steps = lr_scaled_training_steps + lr * num_steps

    return lr_scaled_training_steps


@tf.function
def run_training_step(
    model,
    optimizer_model,
    optimizer_estimator,
    lr_scaled_normalized_training_time,
    x_batch,
    model_loss_coeffs,
    estimator_loss_coeffs,
):
    estimator_weights, model_weights = split_weights(
        model.trainable_weights,
        [
            "probability_estimator",
            "conditional_probability_estimator"
        ],
    )

    with tf.GradientTape() as tape:
        model(x_batch, lr_scaled_normalized_training_time, training=True)
        regularizations = get_ordered_regularization_terms(model)
        model_loss = get_loss(regularizations=regularizations, **model_loss_coeffs)

    grads_model = tape.gradient(model_loss, model_weights)

    with tf.GradientTape() as tape:
        model(x_batch, lr_scaled_normalized_training_time, training=True)
        regularizations = get_ordered_regularization_terms(model)
        estimator_loss = get_loss(regularizations, **estimator_loss_coeffs)

    grads_estimator = tape.gradient(estimator_loss, estimator_weights)

    optimizer_model.apply_gradients(zip(grads_model, model_weights))
    optimizer_estimator.apply_gradients(zip(grads_estimator, estimator_weights))

    return model_loss, regularizations


def log(epoch, step, lr_scaled_normalized_training_time, loss, regularizations):
    loss_term_ids = [
        ro.ALIGNMENT_MAXIMIZATION_REGULARIZATION_ORDER,
        ro.UNIFORMITY_MAXIMIZATION_REGULARIZATION_ORDER,
        ro.MARGINAL_ENTROPY_MINIMIZATION_REGULARIZATION_ORDER,
        ro.JOINT_ENTROPY_MAXIMIZATION_REGULARIZATION_ORDER
    ]
    
    loss_term_descriptions = [
        "alignment",
        "uniformity",
        "h_marginal",
        "h_joint"
    ]

    regularizations = regularizations.numpy()
    loss_term_ids = np.array(loss_term_ids)
    sorting_idxs = list(np.argsort(loss_term_ids))

    msg = "Epoch: " + str(epoch) + " Step: " + str(step) + ", " 
    msg = msg + "Normalized time:{:.2f}".format(lr_scaled_normalized_training_time) + ", "
    msg = msg + "Loss:{:.2f}".format(loss) + ", "

    for i in range(len(loss_term_descriptions)):
        msg = msg + loss_term_descriptions[sorting_idxs[i]] + ":"
        msg = msg + "{:.2f}".format(regularizations[i]) + ", "

    msg = msg[:-2]
    msg = msg + "\r"

    print(msg)


def fit(
    model=None,
    optimizer_model=None,
    optimizer_estimators=None,
    lr_scheduler_model=None,
    lr_scheduler_estimators=None,
    dataset=None,
    batch_size=None,
    epochs=None,
    callbacks=None,
    model_loss_coeffs=None,
    estimator_loss_coeffs=None,
):

    total_lr_scaled_training_steps = compute_total_lr_scaled_training_steps(
        epochs, len(dataset) // batch_size, lr_scheduler_model
    )

    lr_scaled_training_steps = 0.0

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        np.random.seed(epoch)
        np.random.shuffle(dataset)

        lr_model = lr_scheduler_model.get_learning_rate(epoch)
        lr_estimators = lr_scheduler_estimators.get_learning_rate(epoch)

        optimizer_model.learning_rate.assign(lr_model)
        optimizer_estimators.learning_rate.assign(lr_estimators)

        for step in range(len(dataset) // batch_size):
            x_batch = dataset[step * batch_size : (step + 1) * batch_size]

            lr_scaled_training_steps = lr_scaled_training_steps + lr_model
            lr_scaled_normalized_training_time = (
                lr_scaled_training_steps / total_lr_scaled_training_steps
            )

            loss, regularizations = run_training_step(
                model,
                optimizer_model,
                optimizer_estimators,
                lr_scaled_normalized_training_time,
                x_batch,
                model_loss_coeffs,
                estimator_loss_coeffs,
            )

            if step % 20 == 0:
                log(epoch=epoch, 
                    step=step, 
                    lr_scaled_normalized_training_time=lr_scaled_normalized_training_time, 
                    loss=loss, 
                    regularizations=regularizations)

        for callback in callbacks:
            callback.on_epoch_end(epoch)