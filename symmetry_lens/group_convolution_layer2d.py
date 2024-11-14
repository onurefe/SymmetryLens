import tensorflow as tf
from tensorflow import math as tm
from numpy import pi
from symmetry_lens.regularizations import convert_to_regularization_format
from symmetry_lens.probability_estimator import ProbabilityEstimator
from symmetry_lens.conditional_probability_estimator import ConditionalProbabilityEstimator

@tf.keras.utils.register_keras_serializable()
class GroupConvolutionLayer2d(tf.keras.layers.Layer):
    def __init__(
        self,
        n_x_size,
        n_y_size,
        resolution_filter_sigma_decay_tc_in_epochs=10.0,
        resolution_filter_initial_sigma=0.1,
        steps_per_epoch=200,
        zero_padding_size=None,
        use_zero_padding=False,
        name="lifting_layer2d",
        eps=1e-7,
        *args,
        **kwargs
    ):
        self._n_x_size = n_x_size
        self._n_y_size = n_y_size
        self._resolution_filter_sigma_decay_tc_in_epochs = resolution_filter_sigma_decay_tc_in_epochs
        self._resolution_filter_initial_sigma = resolution_filter_initial_sigma
        self._steps_per_epoch = steps_per_epoch
        self._use_zero_padding = use_zero_padding
        self._zero_padding_size = zero_padding_size
        self._eps = eps
        
        super(GroupConvolutionLayer2d, self).__init__(name=name)

    def get_config(self):
        config = super(GroupConvolutionLayer2d, self).get_config()
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape

        if self._zero_padding_size is None:
            self._zero_padding_size = self._n_input_dims
            
        self._create_x_generator_parametrization()
        self._create_y_generator_parametrization()
        self._create_resolution_filter()
        self._create_probability_estimator()
        self._create_conditional_probability_estimator()
        self._create_step_counter()
        
    def call(self, x, lr_scaled_normalized_training_time=None, training=False):
        x = self._demean_and_normalize_input(x)

        # Sample resolution filter and interpolate lifting map.
        resolution_filter = self._sample_resolution_filter()
        lm = self._interpolate_lifting_map(resolution_filter)

        # Lift x.
        y = self._lift(x, lm)

        # Estimate probabilities.
        log_p = self._estimate_pixel_probabilities(y, training=True)
        log_l = self._estimate_pixel_probabilities(y, pixel_shift_x=-1)
        log_r = self._estimate_pixel_probabilities(y, pixel_shift_x=1)
        log_d = self._estimate_pixel_probabilities(y, pixel_shift_y=-1)
        log_u = self._estimate_pixel_probabilities(y, pixel_shift_y=1)
        
        log_p_conditional = self._estimate_conditional_pixel_probabilities(y, training=True)
        log_l_conditional = self._estimate_conditional_pixel_probabilities(y, given_pixel_shift_x=-1, estimated_pixel_shift_x=-1) 
        log_r_conditional = self._estimate_conditional_pixel_probabilities(y, given_pixel_shift_x=1, estimated_pixel_shift_x=1)
        log_d_conditional = self._estimate_conditional_pixel_probabilities(y, given_pixel_shift_y=-1, estimated_pixel_shift_y=-1) 
        log_u_conditional = self._estimate_conditional_pixel_probabilities(y, given_pixel_shift_y=1, estimated_pixel_shift_y=1)
        
        self._increase_step_counter()

        if training:
            self.add_loss(self._compute_alignment_maximization_regularization(y))
            self.add_loss(self._compute_uniformity_maximization_regularization(log_p=log_p, 
                                                                               log_l=log_l, 
                                                                               log_r=log_r, 
                                                                               log_d=log_d,
                                                                               log_u=log_u,
                                                                               log_p_conditional=log_p_conditional, 
                                                                               log_l_conditional=log_l_conditional, 
                                                                               log_r_conditional=log_r_conditional,
                                                                               log_d_conditional=log_d_conditional,
                                                                               log_u_conditional=log_u_conditional))
            
            self.add_loss(self._compute_marginal_entropy_minimization_regularization(log_p))
            self.add_loss(self._compute_joint_entropy_maximization_regularization(y, lr_scaled_normalized_training_time))

        return y

    def _compute_alignment_maximization_regularization(self, y):
        mask = self._get_nbr_mask(max_distance=1.) - self._get_nbr_mask(max_distance=0.)
        
        r = self._cross_correlate(y)
        reg = -tf.reduce_sum(mask * r) / tf.reduce_sum(mask)

        return convert_to_regularization_format("alignment_maximization", reg)

    def _compute_uniformity_maximization_regularization(self,
                                                        log_p, 
                                                        log_l, 
                                                        log_r, 
                                                        log_d,
                                                        log_u,
                                                        log_p_conditional, 
                                                        log_l_conditional, 
                                                        log_r_conditional,
                                                        log_d_conditional,
                                                        log_u_conditional):
        
        reg1 = self._compute_mean_kl_divergence_between_neighbor_pixels(log_p, 
                                                                        log_l, 
                                                                        log_r,
                                                                        log_d,
                                                                        log_u)
        
        reg2 = self._compute_mean_conditional_kl_divergence_between_pixel_pairs(log_p_conditional, 
                                                                                log_l_conditional, 
                                                                                log_r_conditional,
                                                                                log_d_conditional,
                                                                                log_u_conditional)
        
        return convert_to_regularization_format("uniformity_maximization", (reg1 + reg2))

    
    def _compute_marginal_entropy_minimization_regularization(self, log_p):
        h_marginal = self._compute_entropy(log_p)
        reg = tf.reduce_mean(h_marginal)

        return convert_to_regularization_format("marginal_entropy_minimization", reg)

    def _compute_joint_entropy_maximization_regularization(
        self, y, lr_scaled_normalized_training_time
    ):
        normalized_rank = tf.cast(lr_scaled_normalized_training_time, tf.float32)
        h_joint = self._compute_low_rank_entropy(y, normalized_rank)
        reg = -h_joint
        return convert_to_regularization_format("joint_entropy_maximization", reg)
    
    def _compute_mean_conditional_kl_divergence_between_pixel_pairs(self, 
                                                                    log_p_conditional, 
                                                                    log_l_conditional, 
                                                                    log_r_conditional,
                                                                    log_d_conditional,
                                                                    log_u_conditional):
        div_pl_pairs = self._kl_divergence(log_p_conditional, log_l_conditional)
        div_pr_pairs = self._kl_divergence(log_p_conditional, log_r_conditional)
        div_pd_pairs = self._kl_divergence(log_p_conditional, log_d_conditional)
        div_pu_pairs = self._kl_divergence(log_p_conditional, log_u_conditional)
        
        mask_pl_pairs = self._get_conditional_kl_divergence_mask(given_pixel_shift_x=-1, estimated_pixel_shift_x=-1)
        mask_pr_pairs = self._get_conditional_kl_divergence_mask(given_pixel_shift_x=1, estimated_pixel_shift_x=1)
        mask_pd_pairs = self._get_conditional_kl_divergence_mask(given_pixel_shift_y=-1, estimated_pixel_shift_y=-1)
        mask_pu_pairs = self._get_conditional_kl_divergence_mask(given_pixel_shift_y=1, estimated_pixel_shift_y=1)
        
        total_div = (div_pl_pairs * mask_pl_pairs + 
                     div_pr_pairs * mask_pr_pairs + 
                     div_pd_pairs * mask_pd_pairs + 
                     div_pu_pairs * mask_pu_pairs)
        
        mask_sum = mask_pl_pairs + mask_pr_pairs + mask_pd_pairs + mask_pu_pairs
        total_div = total_div / (mask_sum + self._eps)

        num_elements = tf.reduce_sum(tf.where(mask_sum > self._eps, 1.0, 0.0))
        mean_div = tf.reduce_sum(total_div) / num_elements

        return mean_div
    
    def _compute_mean_kl_divergence_between_neighbor_pixels(self, 
                                                            log_p, 
                                                            log_l, 
                                                            log_r,
                                                            log_d,
                                                            log_u):
        div_pl = self._kl_divergence(log_p, log_l)
        div_pr = self._kl_divergence(log_p, log_r)
        div_pd = self._kl_divergence(log_p, log_d)
        div_pu = self._kl_divergence(log_p, log_u)
        
        mask_pl = self._get_kl_divergence_mask(pixel_shift_x=-1)
        mask_pr = self._get_kl_divergence_mask(pixel_shift_x=1)
        mask_pd = self._get_kl_divergence_mask(pixel_shift_y=-1)
        mask_pu = self._get_kl_divergence_mask(pixel_shift_y=1)
        
        total_div = (div_pl * mask_pl + 
                     div_pr * mask_pr + 
                     div_pd * mask_pd + 
                     div_pu * mask_pu)
        
        mask_sum = mask_pl + mask_pr + mask_pd + mask_pu
        total_div = total_div / (mask_sum + self._eps)

        num_elements = tf.reduce_sum(tf.where(mask_sum > self._eps, 1.0, 0.0))
        mean_div = tf.reduce_sum(total_div) / num_elements
        
        return mean_div

    def _estimate_conditional_pixel_probabilities(
        self,
        y,
        given_pixel_shift_x=0,
        estimated_pixel_shift_x=0,
        given_pixel_shift_y=0,
        estimated_pixel_shift_y=0,
        training=False,
    ):
        y_given = tf.roll(tf.roll(y, shift=given_pixel_shift_x, axis=1), shift=given_pixel_shift_y, axis=2)
        y_estimated = tf.roll(tf.roll(y, shift=estimated_pixel_shift_x, axis=1), shift=estimated_pixel_shift_y, axis=2)
        y_given_reshaped = tf.reshape(y_given, shape=[self._batchsize, self._n_x_size * self._n_y_size, 1])
        y_estimated_reshaped = tf.reshape(y_estimated, shape=[self._batchsize, self._n_x_size * self._n_y_size, 1])
        log_p = self._conditional_probability_estimator([y_given_reshaped, y_estimated_reshaped], training=training)
        log_p = tf.reshape(log_p, shape=[-1, self._n_x_size * self._n_y_size, self._n_x_size, self._n_y_size])
        log_p = tf.reshape(log_p, shape=[-1, self._n_x_size, self._n_y_size, self._n_x_size, self._n_y_size])
        
        log_p = tf.roll(log_p,
                        shift=[-given_pixel_shift_x, 
                               -given_pixel_shift_y, 
                               -estimated_pixel_shift_x, 
                               -estimated_pixel_shift_y], 
                        axis=[1, 2, 3, 4])
        
        return log_p
    
    def _estimate_pixel_probabilities(
        self,
        y,
        pixel_shift_x=0,
        pixel_shift_y=0,
        training=False,
    ):
        y = tf.roll(y, shift=[pixel_shift_x, pixel_shift_y], axis=[1, 2])
        
        y_flattened = tf.reshape(y, shape=[self._batchsize, self._n_x_size * self._n_y_size, 1])
        log_p = self._probability_estimator(y_flattened, training=training)
        log_p = tf.reshape(log_p, shape=[self._batchsize, self._n_x_size, self._n_y_size])
        log_p = tf.roll(log_p, shift=[-pixel_shift_x, -pixel_shift_y], axis=[1, 2])

        return log_p
    
    def _get_conditional_kl_divergence_mask(
        self,
        given_pixel_shift_x=0,
        given_pixel_shift_y=0,
        estimated_pixel_shift_x=0,
        estimated_pixel_shift_y=0
    ):
        given_pixel_x_idxs = tf.range(0, self._n_x_size)
        given_pixel_y_idxs = tf.range(0, self._n_y_size)
        estimated_pixel_x_idxs = tf.range(0, self._n_x_size)
        estimated_pixel_y_idxs = tf.range(0, self._n_y_size)
        
        gpi_x = given_pixel_x_idxs + given_pixel_shift_x
        gpi_y = given_pixel_y_idxs + given_pixel_shift_y
        epi_x = estimated_pixel_x_idxs + estimated_pixel_shift_x
        epi_y = estimated_pixel_y_idxs + estimated_pixel_shift_y

        gpi_mask_x = tf.where(
            tf.logical_and((0 <= gpi_x), (gpi_x < self._n_x_size)), 1.0, 0.0
        )
        gpi_mask_y = tf.where(
            tf.logical_and((0 <= gpi_y), (gpi_y < self._n_y_size)), 1.0, 0.0
        )
        epi_mask_x = tf.where(
            tf.logical_and((0 <= epi_x), (epi_x < self._n_x_size)), 1.0, 0.0
        )
        epi_mask_y = tf.where(
            tf.logical_and((0 <= epi_y), (epi_y < self._n_y_size)), 1.0, 0.0
        )
        
        boundary_mask = (gpi_mask_x[:, tf.newaxis, tf.newaxis, tf.newaxis] * 
                         gpi_mask_y[tf.newaxis, :, tf.newaxis, tf.newaxis] * 
                         epi_mask_x[tf.newaxis, tf.newaxis, :, tf.newaxis] * 
                         epi_mask_y[tf.newaxis, tf.newaxis, tf.newaxis, :])

        self_mask = 1.0 - self._get_nbr_mask(max_distance=0.)
        mask = boundary_mask * self_mask
        return mask
    
    def _get_kl_divergence_mask(self, pixel_shift_x=0, pixel_shift_y=0):
        pixel_x_idxs = tf.range(0, self._n_x_size)
        pixel_y_idxs = tf.range(0, self._n_y_size)
        
        shifted_pixel_x_indexes = pixel_x_idxs + pixel_shift_x
        shifted_pixel_y_indexes = pixel_y_idxs + pixel_shift_y
        
        mask_x = tf.where(((0 <= shifted_pixel_x_indexes) & (shifted_pixel_x_indexes < self._n_x_size)), 
                          1.0, 0.0)
        mask_y = tf.where(((0 <= shifted_pixel_y_indexes) & (shifted_pixel_y_indexes < self._n_y_size)), 
                          1.0, 0.0)
        
        mask = mask_x[:, tf.newaxis] * mask_y[tf.newaxis, :]
        return mask
    
    def _lift(self, x, lm):
        y = tf.einsum("bdc, nmd->bnmc", x, lm)
        y = y[:, :, :, 0]
        return y

    def _interpolate_lifting_map(self, resolution_filter):
        filter2px_position_vectors = self._compute_resolution_filter_to_pixel_position_vectors()
        
        if self._use_zero_padding:
            paddings = tf.constant([[self._zero_padding_size, self._zero_padding_size]])
            resolution_filter = tf.pad(resolution_filter, paddings, "CONSTANT")
        
        resolution_filter = resolution_filter[tf.newaxis, tf.newaxis, :]
        resolution_filter_x_propagated = self._propagate_resolution_filter(self._x_generator_parametrization, 
                                                                           resolution_filter, 
                                                                           filter2px_position_vectors[:, :, 0])
        
        resolution_filter_xy_propagated = self._propagate_resolution_filter(self._y_generator_parametrization, 
                                                                            resolution_filter_x_propagated, 
                                                                            filter2px_position_vectors[:, :, 1])
        
        resolution_filter_y_propagated = self._propagate_resolution_filter(self._y_generator_parametrization, 
                                                                           resolution_filter, 
                                                                           filter2px_position_vectors[:, :, 1])
        
        resolution_filter_yx_propagated = self._propagate_resolution_filter(self._x_generator_parametrization, 
                                                                            resolution_filter_y_propagated, 
                                                                            filter2px_position_vectors[:, :, 0])
        weight_xy = tf.random.uniform(shape=[1], minval=0., maxval=1.)
        weight_yx = 1. - weight_xy
        lifting_map = weight_xy * resolution_filter_xy_propagated + weight_yx * resolution_filter_yx_propagated
        
        if self._use_zero_padding:
            lifting_map = lifting_map[:, :, self._zero_padding_size:-self._zero_padding_size]
            
        return lifting_map

    def _propagate_resolution_filter(self, generator_parametrization, resolution_filter, filter2px_position_vector_components):
        log_eigvals, eigvecs = self._decompose_generator_parametrization(generator_parametrization)

        resolution_filter_in_gen_basis = self._map_to_generator_basis(resolution_filter, eigvecs)
        impedances = tf.exp(
            tf.einsum(
                "l, nm->nml", log_eigvals, self._complex(r=filter2px_position_vector_components)
            )
        )
        resolution_filter_propagated_in_gen_basis = resolution_filter_in_gen_basis * impedances

        resolution_filter_propagated = self._retrieve_from_generator_basis(resolution_filter_propagated_in_gen_basis, 
                                                                           eigvecs)

        return resolution_filter_propagated
    
    def _compute_resolution_filter_to_pixel_position_vectors(self):
        resolution_filter_position = tf.convert_to_tensor([0.5 * (self._n_x_size - 1),
                                                           0.5 * (self._n_y_size - 1)])

        pixel_x_positions = tf.linspace(
            start=0.0,
            stop=tf.cast(self._n_x_size - 1, tf.float32),
            num=self._n_x_size,
        )
        
        pixel_y_positions = tf.linspace(
            start=0.0,
            stop=tf.cast(self._n_y_size - 1, tf.float32),
            num=self._n_y_size,
        )
        
        pixel_x_positions = tf.repeat(pixel_x_positions[:, tf.newaxis], repeats=self._n_y_size, axis=-1)
        pixel_y_positions = tf.repeat(pixel_y_positions[tf.newaxis, :], repeats=self._n_x_size, axis=0)
        
        pixel_positions = tf.concat([pixel_x_positions[:, :, tf.newaxis], 
                                     pixel_y_positions[:, :, tf.newaxis]],
                                    axis=-1)
        
        resolution_filter_to_pixel_vectors = (
            pixel_positions - resolution_filter_position[tf.newaxis, tf.newaxis, :]
        )
        return resolution_filter_to_pixel_vectors

    def _increase_step_counter(self):
        incremented_value = self._step_counter + 1
        overflow_mask = tf.cast(incremented_value == 0, tf.uint32)
        next_value = (
            overflow_mask * self._step_counter + (1 - overflow_mask) * incremented_value
        )
        self._step_counter.assign(next_value)
    
    def _decompose_generator_parametrization(self, generator_parametrization):
        s = self._skew_symmetrize(generator_parametrization)
        h = self._complex(i=s)
        eigvals, eigvecs = tf.linalg.eigh(h)
        eigvals = eigvals / self._complex(i=1.0)
        eigvals, eigvecs = self._sort_eigenvalues_and_eigenvectors(eigvals, eigvecs)
        return eigvals, eigvecs

    def _sort_eigenvalues_and_eigenvectors(self, eigenvalues, eigenvectors):
        phases = tf.math.imag(eigenvalues)
        sorter = tf.argsort(phases)
        sorted_eigenvalues = tf.gather(eigenvalues, sorter)
        sorted_eigenvectors = tf.gather(eigenvectors, sorter, axis=1)
        return sorted_eigenvalues, sorted_eigenvectors

    def _cross_correlate(self, x):
        x = x - tf.reduce_mean(x, axis=0, keepdims=True)
        x_l2 = tf.sqrt(tf.reduce_sum(x * x, axis=0) + self._eps)
        x = x / (x_l2 + self._eps)
        r = tf.einsum("bnm, bkl->nkml", x, x)
        
        return r

    def _cross_covariance(self, x):
        x = x - tf.reduce_mean(x, axis=0, keepdims=True)
        r = tf.einsum("bn, bk->nk", x, x) / self._batchsize

        return r

    def _kl_divergence(self, log_p, log_q):
        kl_div = tf.reduce_mean((log_p - log_q), axis=0)
        kl_div = tf.maximum(kl_div, 0.0)

        return kl_div
    
    def _compute_joint_entropy_per_dimension(
        self, y, lr_scaled_normalized_training_time
    ):
        if lr_scaled_normalized_training_time is None:
            lr_scaled_normalized_training_time = 0.
            
        normalized_rank = tf.cast(lr_scaled_normalized_training_time, tf.float32)
        h_joint = self._compute_low_rank_entropy(y, normalized_rank)
        return h_joint

    def _compute_low_rank_entropy(self, z, normalized_rank):
        z_flattened = tf.reshape(z, 
                                 shape=[self._batchsize, self._n_x_size * self._n_y_size])
        cov = self._cross_covariance(z_flattened)
        var = tf.linalg.svd(cov, compute_uv=False)
        h = self._gaussian_entropy_1d(var)
        w = self._soft_thresholding_window(normalized_rank, gain=100, exp_clamp=50)
        low_rank_entropy = tf.reduce_sum(h * w) / tf.reduce_sum(w)

        return low_rank_entropy

    def _compute_equivalent_variance(self, h, d):
        d = tf.cast(d, tf.float32)
        equivalent_variance = tf.sqrt(tf.exp(2. * h / d) / (2. * pi * tf.exp(1.)))
        return equivalent_variance
    
    def _compute_entropy(self, log_p):
        h = -tf.reduce_mean(log_p, axis=0)
        return h

    def _sample_resolution_filter(self):
        sigma = self._compute_resolution_filter_sigma()
        noise = self._generate_resolution_filter_noise()

        resolution_filter_sampled = self._resolution_filter + sigma * noise
        resolution_filter_sampled = self._l2_normalize(resolution_filter_sampled, axis=0)
        return resolution_filter_sampled

    def _generate_resolution_filter_noise(self):
        return tf.random.normal(shape=[self._n_input_dims])

    def _compute_resolution_filter_sigma(self):
        t = tf.cast(self._step_counter, dtype=tf.float32) / self._steps_per_epoch
        sigma = self._resolution_filter_initial_sigma * tf.exp(
            -t / self._resolution_filter_sigma_decay_tc_in_epochs
        )
        return sigma

    def _soft_thresholding_window(self, threshold, gain=100, exp_clamp=50):
        s = tf.linspace(0.0, 1.0, self._n_input_dims)
        exponent = gain * (s - threshold)
        exponent = tf.clip_by_value(
            exponent, clip_value_min=-exp_clamp, clip_value_max=exp_clamp
        )
        return 1.0 / (tf.exp(exponent) + 1.0)

    def _gaussian_entropy_1d(self, var):
        entropy = 0.5 * (tf.math.log(2 * pi * var + self._eps) + 1.0)
        return entropy

    def _map_to_generator_basis(self, m, eigvecs):
        m = tf.expand_dims(m, axis=0)
        m_in_gen_basis = tf.matmul(self._complex(r=m), eigvecs)
        return m_in_gen_basis[0]
    
    def _retrieve_from_generator_basis(self, m_in_gen_basis, eigvecs):
        m = tf.matmul(m_in_gen_basis, tf.linalg.adjoint(eigvecs))
        return tf.math.real(m)
    
    def _demean_and_normalize_input(self, x):
        x = x - tf.reduce_mean(x, axis=(0, 1), keepdims=True)
        x = x / tf.math.reduce_std(x, axis=(0, 1), keepdims=True)
        return x

    def _l2_normalize(self, x, axis):
        norm = tf.sqrt(tf.reduce_sum(x * tf.math.conj(x), axis=axis, keepdims=True))
        return x / (norm + self._eps)

    def _swap_timestep_and_channel_axes(self, x):
        return tf.transpose(x, perm=[0, 3, 1, 2])

    def _skew_symmetrize(self, a):
        a_tr = tf.transpose(a, perm=[1, 0])
        return 0.5 * (a - a_tr)

    def _complex(self, r=0.0, i=0.0):
        return tf.complex(r, i)

    def _get_nbr_mask(self, max_distance):
        positions_x = tf.range(0, self._n_x_size, dtype=tf.float32)
        positions_y = tf.range(0, self._n_y_size, dtype=tf.float32)
        
        x_diff = positions_x[:, tf.newaxis, tf.newaxis, tf.newaxis] - positions_x[tf.newaxis, :, tf.newaxis, tf.newaxis]
        y_diff = positions_y[tf.newaxis, tf.newaxis, :, tf.newaxis] - positions_y[tf.newaxis, tf.newaxis, tf.newaxis, :]
        
        distances = tf.sqrt(tf.square(x_diff) + tf.square(y_diff))
        
        mask = tf.where(distances <= max_distance + self._eps, 1.0, 0.0)
        return mask
    
    def _create_probability_estimator(self):
        y = tf.zeros(shape=[self._batchsize, self._n_x_size * self._n_y_size, 1]) 
        self._probability_estimator = ProbabilityEstimator(name="probability_estimator")
        self._probability_estimator(y)
        
    def _create_conditional_probability_estimator(self):
        y = tf.zeros(shape=[self._batchsize, self._n_x_size * self._n_y_size, 1])
        self._conditional_probability_estimator = ConditionalProbabilityEstimator(num_kernels=4, 
                                                                                  name="conditional_probability_estimator")
        self._conditional_probability_estimator([y, y])
   
    def _create_resolution_filter(self):
        self._resolution_filter = tf.Variable(
            tf.zeros([self._n_input_dims]),
            trainable=True,
            name="resolution_filter",
            dtype=tf.float32,
        )
        
    def _create_x_generator_parametrization(self):
        if self._use_zero_padding:
            initval = tf.random.normal(
                shape=[self._n_input_dims + 2*self._zero_padding_size, 
                       self._n_input_dims + 2*self._zero_padding_size], 
                dtype=tf.float32, 
                stddev=1e-3
            )
        else:
            initval = tf.random.normal(
                shape=[self._n_input_dims, 
                       self._n_input_dims], 
                dtype=tf.float32, 
                stddev=1e-3
            )
            
        self._x_generator_parametrization = tf.Variable(
            initval, trainable=True, name="x_generator_parametrization"
        )
    
    def _create_y_generator_parametrization(self):
        if self._use_zero_padding:
            initval = tf.random.normal(
                shape=[self._n_input_dims + 2*self._zero_padding_size, 
                       self._n_input_dims + 2*self._zero_padding_size], 
                dtype=tf.float32, 
                stddev=1e-3
            )
        else:
            initval = tf.random.normal(
                shape=[self._n_input_dims, 
                       self._n_input_dims], 
                dtype=tf.float32, 
                stddev=1e-3
            )
            
        self._y_generator_parametrization = tf.Variable(
            initval, trainable=True, name="y_generator_parametrization"
        )

    def _create_step_counter(self):
        self._step_counter = tf.Variable(
            tf.zeros([], dtype=tf.uint32), trainable=False, name="step_counter"
        )
        
    @property
    def _lifting_map(self):
        resolution_filter = self._l2_normalize(self._resolution_filter, axis=0)
        return self._interpolate_lifting_map(resolution_filter)

    @property
    def _y_generator(self):
        s = self._skew_symmetrize(self._y_generator_parametrization)
        g = tf.linalg.expm(s)
        return g
    
    @property
    def _x_generator(self):
        s = self._skew_symmetrize(self._x_generator_parametrization)
        g = tf.linalg.expm(s)
        return g

    @property
    def _log_y_generator(self):
        s = self._skew_symmetrize(self._y_generator_parametrization)
        return s
    
    @property
    def _log_x_generator(self):
        s = self._skew_symmetrize(self._x_generator_parametrization)
        return s
    
    @property
    def _n_input_dims(self):
        return self._input_shape[1]
    
    @property
    def _batchsize(self):
        return self._input_shape[0]