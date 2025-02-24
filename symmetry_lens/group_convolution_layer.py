import tensorflow as tf
from tensorflow import math as tm
from numpy import pi
from symmetry_lens.regularizations import convert_to_regularization_format
from symmetry_lens.probability_estimator import ProbabilityEstimator
from symmetry_lens.conditional_probability_estimator import ConditionalProbabilityEstimator

@tf.keras.utils.register_keras_serializable()
class GroupConvolutionLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_uniformity_scales = 1,
        origin_filters_sigma_decay_tc_in_epochs=10.0,
        origin_filters_initial_sigma=0.1,
        num_origin_filters=1,
        steps_per_epoch=200,
        zero_padding_size=None,
        use_zero_padding=False,
        conditional_probability_estimator_hidden_layer_size = None,
        name="group_convolution_layer",
        eps=1e-7,
        *args,
        **kwargs
    ):
        self._num_uniformity_scales = num_uniformity_scales
        self._origin_filters_sigma_decay_tc_in_epochs = origin_filters_sigma_decay_tc_in_epochs
        self._origin_filters_initial_sigma = origin_filters_initial_sigma
        self._conditional_probability_estimator_hidden_layer_size = conditional_probability_estimator_hidden_layer_size
        self._steps_per_epoch = steps_per_epoch
        self._num_origin_filters = num_origin_filters
        self._use_zero_padding = use_zero_padding
        self._zero_padding_size = zero_padding_size
        self._eps = eps
        
        super(GroupConvolutionLayer, self).__init__(name=name)

    def get_config(self):
        config = super(GroupConvolutionLayer, self).get_config()
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape

        if self._zero_padding_size is None:
            self._zero_padding_size = self._n_timesteps
            
        self._create_generator_parametrization()
        self._create_origin_filters()
        self._create_probability_estimators()
        self._create_conditional_probability_estimators()
        self._create_step_counter()
        
    def call(self, x, lr_scaled_normalized_training_time=None, training=False):
        x = self._demean_and_normalize_input(x)

        # Sample origin_filters and interpolate lifting map.
        origin_filters = self._sample_origin_filters()
        lm = self._interpolate_lifting_map(origin_filters)

        # Lift x.
        y = self._lift(x, lm)

        # Estimate probabilities.
        log_p = self._estimate_patch_probabilities(y, training=True)
        log_l = self._estimate_patch_probabilities(y, patch_shift=-1)
        log_r = self._estimate_patch_probabilities(y, patch_shift=1)

        log_p_conditional = self._estimate_conditional_patch_probabilities(y, training=True)
        log_l_conditional = self._estimate_conditional_patch_probabilities(y, given_patch_shift=-1, estimated_patch_shift=-1) 
        log_r_conditional = self._estimate_conditional_patch_probabilities(y, given_patch_shift=1, estimated_patch_shift=1)
        
        self._increase_step_counter()

        if training:
            self.add_loss(self._compute_alignment_maximization_regularization(y))
            self.add_loss(self._compute_uniformity_maximization_regularization(log_p_conditional=log_p_conditional, 
                                                                               log_l_conditional=log_l_conditional, 
                                                                               log_r_conditional=log_r_conditional))
            self.add_loss(self._compute_marginal_entropy_minimization_regularization(log_p))
            self.add_loss(self._compute_joint_entropy_maximization_regularization(y, lr_scaled_normalized_training_time))

        return y

    def _compute_alignment_maximization_regularization(self, y):
        mask = self._nbr_mask_size3 - self._nbr_mask_size1

        r = self._cross_correlate(y)
        reg = -tf.reduce_sum(mask * r) / tf.reduce_sum(mask)

        return convert_to_regularization_format("alignment_maximization", reg)

    def _compute_uniformity_maximization_regularization(self, log_p_conditional, log_l_conditional, log_r_conditional):
        reg = self._compute_mean_conditional_kl_divergence_between_patch_pairs(log_p_conditional, 
                                                                               log_l_conditional, 
                                                                               log_r_conditional)
        
        return convert_to_regularization_format("uniformity_maximization", reg)

    
    def _compute_marginal_entropy_minimization_regularization(self, log_p):
        h_marginal = self._compute_entropy(log_p[0])
        reg = tf.reduce_mean(h_marginal)

        return convert_to_regularization_format("marginal_entropy_minimization", reg)

    def _compute_joint_entropy_maximization_regularization(
        self, y, lr_scaled_normalized_training_time
    ):
        normalized_rank = tf.cast(lr_scaled_normalized_training_time, tf.float32)
        h_joint = self._compute_low_rank_entropy(y, normalized_rank)
        reg = -h_joint
        return convert_to_regularization_format("joint_entropy_maximization", reg)
    
    def _compute_mean_conditional_kl_divergence_between_patch_pairs(self, log_p_conditional, log_l_conditional, log_r_conditional):
        statistical_distances = []

        for scale_idx in range(self._num_uniformity_scales):
            div_former_pairs = self._kl_divergence(log_p_conditional[scale_idx], log_l_conditional[scale_idx])
            div_next_pairs = self._kl_divergence(log_p_conditional[scale_idx], log_r_conditional[scale_idx])

            mask_former_pairs = self._get_conditional_kl_divergence_mask(scale_idx, given_patch_shift=-1, estimated_patch_shift=-1)
            mask_next_pairs = self._get_conditional_kl_divergence_mask(scale_idx, given_patch_shift=1, estimated_patch_shift=1)

            div_bidir = div_former_pairs * mask_former_pairs + div_next_pairs * mask_next_pairs
            div_bidir = div_bidir / (mask_former_pairs + mask_next_pairs + self._eps)

            num_elements = tf.reduce_sum(tf.where(mask_former_pairs + mask_next_pairs > self._eps, 1.0, 0.0))
            mean_div = tf.reduce_sum(div_bidir) / num_elements

            normalized_div = mean_div / tf.cast(self._get_patch_size(scale_idx), tf.float32)
            statistical_distances.append(normalized_div)

        mean_statistical_distance = tf.reduce_mean(tf.convert_to_tensor(statistical_distances))
        return mean_statistical_distance
    
    def _compute_mean_kl_divergence_between_neighbor_patches(self, log_p, log_l, log_r):
        statistical_distances = []

        for scale_idx in range(self._num_uniformity_scales):
            if scale_idx > 0:
                log_pr_cond = log_p[scale_idx] - log_p[scale_idx-1][:, :-1]
                log_pl_cond = log_p[scale_idx] - log_p[scale_idx-1][:, 1:]
                log_l_cond = log_l[scale_idx] - log_l[scale_idx-1][:, 1:]
                log_r_cond = log_r[scale_idx] - log_r[scale_idx-1][:, :-1]
            else:
                log_pr_cond = log_p[scale_idx]
                log_pl_cond = log_p[scale_idx]
                log_l_cond = log_l[scale_idx]
                log_r_cond = log_r[scale_idx]
                
            div_former = self._kl_divergence(log_pl_cond, log_l_cond)
            div_next = self._kl_divergence(log_pr_cond, log_r_cond)
            
            mask_former = self._get_kl_divergence_mask(scale_idx, patch_shift=-1)
            mask_next = self._get_kl_divergence_mask(scale_idx, patch_shift=1)

            div_bidir = div_former * mask_former + div_next * mask_next
            div_bidir = div_bidir / (mask_former + mask_next + self._eps)

            num_elements = tf.reduce_sum(tf.where(mask_former + mask_next > self._eps, 1.0, 0.0))
            mean_kl_div = tf.reduce_sum(div_bidir) / num_elements

            normalized_div = mean_kl_div / tf.cast(self._get_patch_size(scale_idx), tf.float32)
            statistical_distances.append(normalized_div)

        mean_statistical_distance = tf.reduce_mean(tf.convert_to_tensor(statistical_distances))
        return mean_statistical_distance

    def _estimate_conditional_patch_probabilities(
        self,
        y,
        given_patch_shift=0,
        estimated_patch_shift=0,
        training=False,
    ):
        patch_probabilities = []

        for scale_idx in range(self._num_uniformity_scales):
            y_patches = self._form_patches(y, scale_idx)
            y_given = tf.roll(y_patches, shift=given_patch_shift, axis=1)
            y_estimated = tf.roll(y_patches, shift=estimated_patch_shift, axis=1)
            p = self._conditional_probability_estimators[scale_idx]([y_given, y_estimated], training=training)
            p = tf.roll(p, shift=[-given_patch_shift, -estimated_patch_shift], axis=[1, 2])
            patch_probabilities.append(p)

        return patch_probabilities
    
    def _estimate_patch_probabilities(
        self,
        y,
        patch_shift=0,
        training=False,
    ):
        patch_probabilities = []

        for scale_idx in range(self._num_uniformity_scales):
            y_patches = self._form_patches(y, scale_idx)
            y_patches = tf.roll(y_patches, shift=patch_shift, axis=1)
            log_p = self._probability_estimators[scale_idx](y_patches, training=training)
            log_p = tf.roll(log_p, shift=-patch_shift, axis=1)
            patch_probabilities.append(log_p)

        return patch_probabilities
    
    def _get_conditional_kl_divergence_mask(
        self,
        scale_idx,
        given_patch_shift=0,
        estimated_patch_shift=0,
    ):
        num_patches = self._get_num_patches(scale_idx)

        given_patch_idxs = tf.range(0, num_patches)
        estimated_patch_idxs = tf.range(0, num_patches)

        gpi = given_patch_idxs + given_patch_shift
        epi = estimated_patch_idxs + estimated_patch_shift

        gpi_mask = tf.where(
            tf.logical_and((0 <= gpi), (gpi < num_patches)), 1.0, 0.0
        )
        epi_mask = tf.where(
            tf.logical_and((0 <= epi), (epi < num_patches)), 1.0, 0.0
        )
        diag_mask = 1.0 - tf.eye(num_patches)

        mask = gpi_mask[:, tf.newaxis] * epi_mask[tf.newaxis, :] * diag_mask
        return mask
    
    def _get_kl_divergence_mask(self, scale_idx, patch_shift=0):
        num_patches = self._get_num_patches(scale_idx)
        
        patch_indexes = tf.range(0, num_patches)
        shifted_patch_indexes = patch_indexes + patch_shift
        mask = tf.where(((0 <= shifted_patch_indexes) & (shifted_patch_indexes < num_patches)), 1.0, 0.0)

        return mask
    
    def _lift(self, x, lm):
        x = self._swap_timestep_and_channel_axes(x)
        y = tf.matmul(x, lm)
        y = self._swap_timestep_and_channel_axes(y)
        return y

    def _interpolate_lifting_map(self, origin_filters, decay_rate=0.):
        anc2px_position_vectors = self._compute_origin_filter_to_pixel_position_vectors()
        origin_filters_propagated = self._propagate_origin_filters(origin_filters, anc2px_position_vectors)
        weights = tf.exp(-decay_rate * tf.abs(anc2px_position_vectors))
        weights = weights / tf.reduce_sum(weights, axis=0, keepdims=True)
        lifting_map = tf.einsum("anl, an->ln", origin_filters_propagated, weights)
        return lifting_map

    def _propagate_origin_filters(self, origin_filters, anc2px_position_vectors):
        log_eigvals, eigvecs = self._decompose_generator_parametrization()
        
        if self._use_zero_padding:
            paddings = tf.constant([[0, 0], [self._zero_padding_size, self._zero_padding_size]])
            origin_filters = tf.pad(origin_filters, paddings, "CONSTANT")
        
        powers = anc2px_position_vectors
        origin_filters_in_genb = self._map_to_generator_basis(origin_filters, eigvecs)
        impedances = tf.exp(
            tf.einsum(
                "an, l->anl", self._complex(r=powers), log_eigvals
            )
        )
        origin_filters_propagated_in_genb = tf.einsum(
            "al, anl->anl", origin_filters_in_genb, impedances
        )
        propagated_origin_filters = self._retrieve_from_generator_basis(origin_filters_propagated_in_genb, eigvecs)

        if self._use_zero_padding:
            propagated_origin_filters = propagated_origin_filters[:, :, self._zero_padding_size:-self._zero_padding_size]

        return propagated_origin_filters
    
    def _compute_origin_filter_to_pixel_position_vectors(self):
        if self._num_origin_filters > 1:
            origin_filter_positions = tf.linspace(
                start=0.0,
                stop=tf.cast(self._n_timesteps - 1, tf.float32),
                num=self._num_origin_filters,
            )
        else:
            origin_filter_positions = tf.convert_to_tensor([0.5 * (self._n_timesteps - 1)])

        pixel_positions = tf.linspace(
            start=0.0,
            stop=tf.cast(self._n_timesteps - 1, tf.float32),
            num=self._n_timesteps,
        )
        origin_filter_to_pixel_vectors = (
            pixel_positions[tf.newaxis, :] - origin_filter_positions[:, tf.newaxis]
        )
        return origin_filter_to_pixel_vectors

    def _increase_step_counter(self):
        incremented_value = self._step_counter + 1
        overflow_mask = tf.cast(incremented_value == 0, tf.uint32)
        next_value = (
            overflow_mask * self._step_counter + (1 - overflow_mask) * incremented_value
        )
        self._step_counter.assign(next_value)

    def _decompose_generator_parametrization(self):
        s = self._skew_symmetrize(self._generator_parametrization)
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
        r = tf.einsum("bni, bmi->nmi", x, x)
        r = tf.reduce_sum(r, axis=-1)

        return r

    def _cross_covariance(self, x):
        x = x - tf.reduce_mean(x, axis=0, keepdims=True)
        r = tf.einsum("bni, bmi->nmi", x, x) / self._batchsize
        r = tf.reduce_sum(r, axis=-1)

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
        cov = self._cross_covariance(z)
        var = tf.linalg.svd(cov, compute_uv=False)
        h = self._gaussian_entropy_1d(var)
        w = self._soft_thresholding_window(normalized_rank)
        low_rank_entropy = tf.reduce_sum(h * w) / tf.reduce_sum(w)

        return low_rank_entropy

    def _compute_equivalent_variance(self, h, d):
        d = tf.cast(d, tf.float32)
        equivalent_variance = tf.sqrt(tf.exp(2. * h / d) / (2. * pi * tf.exp(1.)))
        return equivalent_variance
    
    def _compute_entropy(self, log_p):
        h = -tf.reduce_mean(log_p, axis=0)
        return h

    def _sample_origin_filters(self):
        sigma = self._compute_origin_filters_sigma()
        noise = self._generate_origin_filters_noise()

        origin_filters_sampled = self._origin_filters + sigma * noise
        origin_filters_sampled = self._l2_normalize(origin_filters_sampled, axis=1)
        return origin_filters_sampled

    def _generate_origin_filters_noise(self):
        return tf.random.normal(shape=[self._num_origin_filters, self._n_timesteps])

    def _compute_origin_filters_sigma(self):
        t = tf.cast(self._step_counter, dtype=tf.float32) / self._steps_per_epoch
        sigma = self._origin_filters_initial_sigma * tf.exp(
            -t / self._origin_filters_sigma_decay_tc_in_epochs
        )
        return sigma

    def _soft_thresholding_window(self, threshold, gain=50, exp_clamp=50):
        s = tf.linspace(0.0, 1.0, self._n_timesteps)
        exponent = gain * (s - threshold)
        exponent = tf.clip_by_value(
            exponent, clip_value_min=-exp_clamp, clip_value_max=exp_clamp
        )
        return 1.0 / (tf.exp(exponent) + 1.0)

    def _gaussian_entropy_1d(self, var):
        entropy = 0.5 * (tf.math.log(2 * pi * var + self._eps) + 1.0)
        return entropy

    def _map_to_generator_basis(self, m, eigvecs):
        return tf.matmul(self._complex(r=m), eigvecs)

    def _retrieve_from_generator_basis(self, m_in_gen_basis, eigvecs):
        m = tf.matmul(m_in_gen_basis, tf.linalg.adjoint(eigvecs))
        return tf.math.real(m)

    def _frobenius_norm(self, x):
        x2 = tm.abs(x * x)
        return tf.sqrt(tf.reduce_sum(x2, axis=(0, 1)) + self._eps)

    def _form_patches(self, y, scale_idx):
        patch_size = self._get_patch_size(scale_idx)
        y = tf.signal.frame(y, patch_size, 1, axis=1)
        num_patches = self._get_num_patches(scale_idx)
        y = tf.reshape(y, shape=[self._batchsize, 
                                 num_patches, 
                                 self._n_channels * patch_size])
        return y
    
    def _get_patch_size(self, scale_idx):
        return scale_idx + 1
        
    def _get_num_patches(self, scale_idx):
        num_patches = 1 + self._n_timesteps - self._get_patch_size(scale_idx)
        return num_patches
    
    def _demean_and_normalize_input(self, x):
        x = x - tf.reduce_mean(x, axis=(0, 1), keepdims=True)
        x = x / tf.math.reduce_std(x, axis=(0, 1), keepdims=True)
        return x

    def _l2_normalize(self, x, axis):
        norm = tf.sqrt(tf.reduce_sum(x * tf.math.conj(x), axis=axis, keepdims=True))
        return x / (norm + self._eps)
    
    def _demean(self, x, axis):
        x = x - tf.reduce_mean(x, axis=axis, keepdims=True)
        return x

    def _normalize(self, x, axis):
        x = x / tf.math.reduce_std(x, axis=axis, keepdims=True)
        return x

    def _swap_timestep_and_channel_axes(self, x):
        return tf.transpose(x, perm=[0, 2, 1])

    def _skew_symmetrize(self, a):
        a_tr = tf.transpose(a, perm=[1, 0])
        return 0.5 * (a - a_tr)

    def _complex(self, r=0.0, i=0.0):
        return tf.complex(r, i)

    def _create_probability_estimators(self):
        self._probability_estimators = []
        
        for scale_idx in range(0, self._num_uniformity_scales):
            pe = ProbabilityEstimator(name="probability_estimator_{}".format(scale_idx))
            patch_size = self._get_patch_size(scale_idx)
            num_patches = self._get_num_patches(scale_idx)
            y = tf.zeros(shape=[self._batchsize, num_patches, patch_size]) 
            pe(y)
            self._probability_estimators.append(pe)

    def _create_conditional_probability_estimators(self):
        self._conditional_probability_estimators = []
        
        for scale_idx in range(0, self._num_uniformity_scales):
            patch_size = self._get_patch_size(scale_idx)
            pe = ConditionalProbabilityEstimator(num_kernels=4, 
                                                 hidden_layer_size=self._conditional_probability_estimator_hidden_layer_size,
                                                 name="conditional_probability_estimator_{}".format(scale_idx))
            
            num_patches = self._get_num_patches(scale_idx)
            y = tf.zeros(shape=[self._batchsize, num_patches, patch_size])
            pe([y, y])
            self._conditional_probability_estimators.append(pe)
            
    def _create_origin_filters(self):
        self._origin_filters = tf.Variable(
            tf.zeros([self._num_origin_filters, self._n_timesteps]),
            trainable=True,
            name="origin_filters",
            dtype=tf.float32,
        )
        
    def _create_generator_parametrization(self):
        if self._use_zero_padding:
            initval = tf.random.normal(
                shape=[self._n_timesteps + 2*self._zero_padding_size, 
                       self._n_timesteps + 2*self._zero_padding_size], 
                dtype=tf.float32, 
                stddev=1e-3
            )
        else:
            initval = tf.random.normal(
                shape=[self._n_timesteps, 
                       self._n_timesteps], 
                dtype=tf.float32, 
                stddev=1e-3
            )
            
        self._generator_parametrization = tf.Variable(
            initval, trainable=True, name="generator_parametrization"
        )

    def _create_step_counter(self):
        self._step_counter = tf.Variable(
            tf.zeros([], dtype=tf.uint32), trainable=False, name="step_counter"
        )
        
    @property
    def _lifting_map(self):
        origin_filters = self._l2_normalize(self._origin_filters, axis=1)
        return self._interpolate_lifting_map(origin_filters)

    @property
    def _generator(self):
        s = self._skew_symmetrize(self._generator_parametrization)
        g = tf.linalg.expm(s)
        return g

    @property
    def _log_generator(self):
        s = self._skew_symmetrize(self._generator_parametrization)
        return s

    @property
    def _nbr_mask_size1(self):
        return tf.eye(self._n_timesteps, dtype=tf.float32)

    @property
    def _nbr_mask_size3(self):
        positions = tf.range(0, self._n_timesteps, dtype=tf.float32)
        distances = tf.abs(positions[:, tf.newaxis] - positions[tf.newaxis, :])
        mask = tf.where(distances <= 1, 1.0, 0.0)
        
        return mask
    
    @property
    def _n_channels(self):
        return self._input_shape[2]

    @property
    def _n_timesteps(self):
        return self._input_shape[1]

    @property
    def _batchsize(self):
        return self._input_shape[0]