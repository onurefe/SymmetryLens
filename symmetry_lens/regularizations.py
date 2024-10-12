import tensorflow as tf

ALIGNMENT_MAXIMIZATION_REGULARIZATION_ORDER = 1
UNIFORMITY_MAXIMIZATION_REGULARIZATION_ORDER = 2
MARGINAL_ENTROPY_MINIMIZATION_REGULARIZATION_ORDER = 3
JOINT_ENTROPY_MAXIMIZATION_REGULARIZATION_ORDER = 4
PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER = 5
CONDITIONAL_PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER = 6

REGULARIZATIONS = {
    "alignment_maximization": {"order": ALIGNMENT_MAXIMIZATION_REGULARIZATION_ORDER},
    "uniformity_maximization": {"order": UNIFORMITY_MAXIMIZATION_REGULARIZATION_ORDER},
    "marginal_entropy_minimization": {
        "order": MARGINAL_ENTROPY_MINIMIZATION_REGULARIZATION_ORDER
    },
    "joint_entropy_maximization": {"order": JOINT_ENTROPY_MAXIMIZATION_REGULARIZATION_ORDER},
    "probability_estimator_entropy_minimization": {
        "order": PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER
    },
    "conditional_probability_estimator_entropy_minimization": {
        "order": CONDITIONAL_PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER
    }
}

def convert_to_regularization_format(regularization_key, value):
    order = REGULARIZATIONS[regularization_key]["order"]
    return tf.convert_to_tensor([order, value], dtype=tf.float32)