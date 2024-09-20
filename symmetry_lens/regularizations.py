import tensorflow as tf

class RegularizationOrders():
    ALIGNMENT_MAXIMIZATION_REGULARIZATION_ORDER = 1
    UNIFORMITY_MAXIMIZATION_REGULARIZATION_ORDER = 2
    MARGINAL_ENTROPY_MINIMIZATION_REGULARIZATION_ORDER = 3
    JOINT_ENTROPY_MAXIMIZATION_REGULARIZATION_ORDER = 4
    PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER = 5
    CONDITIONAL_PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER = 6

    def __init__(self):
        pass

def get_regularization_dictionary():
    REGULARIZATIONS = {
        "alignment_maximization": {"order": RegularizationOrders.ALIGNMENT_MAXIMIZATION_REGULARIZATION_ORDER},
        "uniformity_maximization": {"order": RegularizationOrders.UNIFORMITY_MAXIMIZATION_REGULARIZATION_ORDER},
        "marginal_entropy_minimization": {
            "order": RegularizationOrders.MARGINAL_ENTROPY_MINIMIZATION_REGULARIZATION_ORDER
        },
        "joint_entropy_maximization": {"order": RegularizationOrders.JOINT_ENTROPY_MAXIMIZATION_REGULARIZATION_ORDER},
        "probability_estimator_entropy_minimization": {
            "order": RegularizationOrders.PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER
        },
        "conditional_probability_estimator_entropy_minimization": {
            "order": RegularizationOrders.CONDITIONAL_PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER
        }
    }    
        
    return REGULARIZATIONS


def convert_to_regularization_format(regularization_key, value):
    regularizations = get_regularization_dictionary()
    order = regularizations[regularization_key]["order"]
    return tf.convert_to_tensor([order, value], dtype=tf.float32)