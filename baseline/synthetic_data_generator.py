import numpy as np
from scipy import special
import tensorflow as tf

class SyntheticDataGenerator:
    def __init__(
        self,
        batch_size,
        features,
        use_circulant_translations=False,
        waveform_timesteps=33,
        noise_normalized_std=0.0,
        output_representation="natural",
        seismic_dataset_path=None,
        p_exist = 0.5,
        num_of_lots = 5,
        eps=1e-6,
    ):
        self._batch_size = batch_size
        self._features = features
        self._use_circulant_translations = use_circulant_translations
        self._waveform_timesteps = waveform_timesteps
        self._noise_normalized_std = noise_normalized_std
        self._output_representation = output_representation
        self._p_exist = p_exist / len(features)
        self._num_of_lots = num_of_lots
        self._eps = eps
        
        (x_train, __), (x_test, __) = tf.keras.datasets.mnist.load_data()
        self._mnist_data = np.concatenate([x_train, x_test], axis=0)
        
        if seismic_dataset_path is not None:
            self._seismic_dataset = np.load(seismic_dataset_path)
            self._idx = 0
            
    def sample_batch_of_data(self):
        x = np.zeros(shape=[self._batch_size, self._waveform_timesteps], dtype=np.float32)
        
        for feature in self._features:
            logits = self._sample_batch_logits() 
            num_samples = self._batch_size * self._num_of_lots
            
            if self._use_circulant_translations:
                center_min = self.timestep_values[0]
                center_max = self.timestep_values[-1]
            else:
                center_min = -self._waveform_timesteps
                center_max = self._waveform_timesteps
            
            if feature["type"] == "gaussian":
                x_feature = self.lot_gaussians(num_samples,
                                                center_min=center_min, 
                                                center_max=center_max, 
                                                sigma_min=feature["scale_min"],
                                                sigma_max=feature["scale_max"],
                                                amplitude_min=feature["amplitude_min"],
                                                amplitude_max=feature["amplitude_max"])
            elif feature["type"] == "mnist_slices":
                total_length = 0
                samples = []
                while True:
                    if self._use_circulant_translations:
                        x_sampled = self._lot_mnist_crops_circulantly(radius=10.5)
                    else:
                        x_sampled = self._lot_mnist_crops()
                    
                    x_sampled = self._filter_blank_images(x_sampled)
                    samples.append(x_sampled)
                    total_length = total_length + len(x_sampled)
                    if total_length >= self._batch_size:
                        break
                    
                x = tf.concat(samples, axis=0)
                x = x[0: self._batch_size]
                break
            elif feature["type"] == "legendre":
                x_feature = self.lot_legendre_polynomials(num_samples,
                                                          center_min=center_min, 
                                                          center_max=center_max, 
                                                          length_min=feature["scale_min"],
                                                          length_max=feature["scale_max"],
                                                          amplitude_min=feature["amplitude_min"],
                                                          amplitude_max=feature["amplitude_max"],
                                                          l=feature["l"],
                                                          m=feature["m"]) 
            else:
                raise ValueError('Feature type is not defined.')
                
            x_feature = np.reshape(x_feature, newshape=[self._batch_size, self._num_of_lots, self._waveform_timesteps])
            x_feature = np.sum(x_feature * (logits[:, :, np.newaxis]), axis=1)
            x = x + x_feature
        
        x = self._demean(x, axis=(0 ,1))
        x = self._std_normalize(x, axis=(0, 1))                
        x = self._add_noise(x)

        x = np.expand_dims(x, axis=-1)

        if self._output_representation == "natural":
            x_final = x
        if self._output_representation == "permuted":
            x_final = self._apply_permutation_map(x)
        elif self._output_representation == "dst":
            x_final = self._apply_dst_transform(x)

        return x_final

    def _filter_blank_images(self, x_images):
        image_maximums = tf.reduce_max(x_images, axis=(1))
        mask = image_maximums > self._eps
        x_images = tf.boolean_mask(x_images, mask)
    
        return x_images
    
    def _lot_mnist_crops(self):
        # --------------------------------------------------
        # 1) Gather and pad the MNIST images
        # --------------------------------------------------
        random_indices = tf.random.uniform(
            shape=[self._batch_size],
            minval=0,
            maxval=len(self._mnist_data),
            dtype=tf.int32
        )
        
        mnist_x_size = tf.shape(self._mnist_data)[1]
        mnist_y_size = tf.shape(self._mnist_data)[2]
        
        sampled_images = tf.gather(self._mnist_data, random_indices)  # (B, 28, 28)
        sampled_images = tf.expand_dims(sampled_images, axis=-1)   # (B, 28, 28, 1)

        max_random_translation = mnist_x_size
        mnist_x_padded_size = mnist_x_size + 2 * max_random_translation
        
        padded_images = tf.image.pad_to_bounding_box(
            sampled_images,
            offset_height=0,
            offset_width=max_random_translation,
            target_height=mnist_y_size,
            target_width=mnist_x_padded_size
        )

        # --------------------------------------------------
        # 2) Define one bounding box for each image
        # --------------------------------------------------
        height_ratio = 1.
        width_ratio  = tf.cast(self._waveform_timesteps, tf.float32) / tf.cast(mnist_x_padded_size, tf.float32)

        x1 = tf.random.uniform([self._batch_size], 0.0, 1.0 - width_ratio)
        y1 = tf.random.uniform([self._batch_size], 0.0, 1.0 - height_ratio)
        
        x2 = x1 + width_ratio
        y2 = y1 + height_ratio

        boxes = tf.stack([y1, x1, y2, x2], axis=1)

        # --------------------------------------------------
        # 3) Crop & Resize
        # --------------------------------------------------
        box_indices = tf.range(self._batch_size)
        
        # Size format is [crop_height, crop_width].
        crop_size = [1, self._waveform_timesteps]

        mnist_cropped = tf.image.crop_and_resize(
            padded_images,
            boxes,
            box_indices,
            crop_size
        )
        
        mnist_cropped = tf.reshape(
            mnist_cropped,
            [self._batch_size, self._waveform_timesteps]
        )
        
        return mnist_cropped
    
    def _lot_mnist_crops_circulantly(self, radius):
        # --------------------------------------------------
        # 1) Gather and pad the MNIST images
        # --------------------------------------------------
        random_indices = tf.random.uniform(
            shape=[self._batch_size],
            minval=0,
            maxval=len(self._mnist_data),
            dtype=tf.int32
        )
        
        mnist_x_size = tf.shape(self._mnist_data)[2]
        mnist_y_size = tf.shape(self._mnist_data)[1]
        
        sampled_images = tf.gather(self._mnist_data, random_indices)  # (B, 28, 28)
        sampled_images = tf.expand_dims(sampled_images, axis=-1)   # (B, 28, 28, 1)

        x_idxs = tf.range(0, mnist_x_size, dtype=tf.float32)
        y_idxs = tf.range(0, mnist_y_size, dtype=tf.float32)
        grid_x_vals, grid_y_vals = tf.meshgrid(x_idxs, y_idxs)
    
        angles = tf.linspace(start=0., stop=2*np.pi, num=(self._waveform_timesteps+1))[:-1]
        angles = angles[tf.newaxis, :] + tf.random.uniform([self._batch_size, 1], minval=-np.pi, maxval=np.pi)
        
        centers_x = 0.5 * tf.cast(mnist_x_size, tf.float32) + tf.cast(radius, tf.float32) * tf.math.cos(angles)
        centers_y = 0.5 * tf.cast(mnist_y_size, tf.float32) + tf.cast(radius, tf.float32) * tf.math.sin(angles)
        
        delta_x = centers_x[:, :, tf.newaxis, tf.newaxis] - grid_x_vals[tf.newaxis, tf.newaxis, :, :]
        delta_y = centers_y[:, :, tf.newaxis, tf.newaxis] - grid_y_vals[tf.newaxis, tf.newaxis, :, :]
        
        sigma_squared = tf.square(centers_x[0, 1] - centers_x[0, 0]) + tf.square(centers_y[0, 1] - centers_y[0, 0])
        weights = tf.exp(-0.5 * (tf.square(delta_x) + tf.square(delta_y)) / sigma_squared)
        weights = weights / tf.reduce_sum(weights, axis=(2, 3), keepdims=True)
        
        interpolated_images = tf.einsum("bhwc, bdhw->bd", 
                                        tf.cast(sampled_images,tf.float32), 
                                        weights)
        
        return interpolated_images
    
    def _get_batch_of_waveforms(self):
        x = self._seismic_dataset[self._idx * self.batch_size:(self._idx+1) * self.batch_size]
        self._idx = self._idx + 1
        return x
        
    def lot_gaussians(self, num_samples, center_min, center_max, sigma_min, sigma_max, amplitude_min, amplitude_max):
        centers = np.random.uniform(center_min, center_max, size=[num_samples])
        sigmas = np.random.uniform(sigma_min, sigma_max, size=[num_samples])
        amplitude = np.random.uniform(amplitude_min, amplitude_max, size=[num_samples])
        centers = np.expand_dims(centers, axis=-1)
        sigmas = np.expand_dims(sigmas, axis=-1)
        amplitudes = np.expand_dims(amplitude, axis=-1)
        t = np.expand_dims(self.timestep_values, axis=0)
        t = np.repeat(t, num_samples, axis=0)
        x = amplitudes * self._gaussian(t, centers, sigmas)
        
        return x

    def lot_dispersed_gaussian(self, 
                               num_samples, 
                               center_min, 
                               center_max, 
                               sigma_min, 
                               sigma_max, 
                               amplitude_min, 
                               amplitude_max, 
                               phase_velocity_coeffs=[1., 0., 0.]):
        centers = np.random.uniform(center_min, center_max, size=[num_samples])
        sigmas = np.random.uniform(sigma_min, sigma_max, size=[num_samples])
        amplitudes = np.random.uniform(amplitude_min, amplitude_max, size=[num_samples])
        centers = np.expand_dims(centers, axis=-1)
        sigmas = np.expand_dims(sigmas, axis=-1)
        amplitudes = np.expand_dims(amplitudes, axis=-1)
        t = np.expand_dims(self.timestep_values, axis=0)
        x = amplitudes * self._dispersed_gaussian(t, centers, sigmas, phase_velocity_coeffs)
        return x
    
    def lot_legendre_polynomials(self, num_samples, center_min, center_max, length_min, length_max, amplitude_min, amplitude_max, l, m):
        centers = np.random.uniform(center_min, center_max, size=[num_samples])
        lengths = np.random.uniform(length_min, length_max, size=[num_samples])
        amplitude = np.random.uniform(amplitude_min, amplitude_max, size=[num_samples])
        centers = np.expand_dims(centers, axis=-1)
        lengths = np.expand_dims(lengths, axis=-1)
        amplitudes = np.expand_dims(amplitude, axis=-1)
        t = np.expand_dims(self.timestep_values, axis=0)
        t = np.repeat(t, num_samples, axis=0)
        
        if self._use_circulant_translations:
            tdiff = self._circulant_position_diff(t, centers)
        else:
            tdiff = t - centers
            
        t_relative_scaled = tdiff / (lengths * 0.5)
        t_relative_scaled_clipped = np.clip(t_relative_scaled, a_min=-1.0, a_max=1.0)
        
        x = amplitudes * self._assoc_legendre_reparam_func(t_relative_scaled_clipped, l, m)            
        return x

    def _dispersed_gaussian(self, t, centers, sigmas, phase_velocity_coeffs, num_freqs_for_propagation_estimation=1000):   
        # Initial wavefunction (Gaussian pulse)
        dt = t[0, 1] - t[0, 0]
        fs = 1./dt
        
        # Convert gaussian to a frequency domain gaussian for Monte Carlo estimation.
        f = np.random.uniform(0, fs/2, size=[1, num_freqs_for_propagation_estimation//2])
        f = np.concatenate([-f[:, ::-1], f], axis=1)
        f_sigmas = (1.0 / (np.pi * sigmas))
        psi_0_f = np.exp(-np.square(f) / (2 * f_sigmas**2))
        psi_0_f = psi_0_f / (np.sqrt(2 * np.pi) * f_sigmas)
        
        # Dispersion relation
        u = self._phase_velocity(f, dt, phase_velocity_coeffs)
        
        # Precompute phase factors for each sample, frequency
        phases = 2 * np.pi * (f / u) * centers

        # Compute psi propagated.
        psi_propageted_f = psi_0_f * np.exp(1j * phases)
        
        # Frequency domain inversion tensor.
        inversion_tensor = np.exp(1j * 2 * np.pi * f[:, :, np.newaxis] * t[np.newaxis, :])
        psi_propagated = np.matmul(psi_propageted_f, inversion_tensor[0])
        psi_propagated = np.real(psi_propagated)

        return psi_propagated

    def _phase_velocity(self, f, dt, phase_velocity_coeffs):
        f_sampling = 1./dt
        f_nyquist = f_sampling/2
        f_abs = np.abs(f)
        f_abs_normalized = f_abs / f_nyquist
        
        result = np.zeros_like(f_abs_normalized)
        for i, phase_velocity_coeff in enumerate(phase_velocity_coeffs):
            result = result + phase_velocity_coeff * np.power(f_abs_normalized, i)
            
        return result

    def _apply_permutation_map(self, x):
        x = np.transpose(x, axes=[0, 2, 1])
        x = np.matmul(x, self.permutation_matrix)
        x = np.transpose(x, axes=[0, 2, 1])
        return x

    def _apply_dst_transform(self, x):
        x = np.transpose(x, axes=[0, 2, 1])
        x = np.matmul(x[:, :, None, :], self.dst_matrix)
        x = np.transpose(x, axes=[0, 3, 1, 2])
        return x[..., 0]

    def _add_noise(self, x_batch):
        x_batch_std = np.std(x_batch, axis=(0, 1), dtype=np.float32, keepdims=True)
        noise = (
            self._noise_normalized_std
            * x_batch_std
            * np.random.normal(0.0, 1, size=np.shape(x_batch))
        )
        return x_batch + noise.astype(np.float32)

    def _sample_batch_logits(self):
        sample_from_uniform_distribution = np.random.uniform(0.0, 1.0, size=[self._batch_size, self._num_of_lots])
        logits = np.where(sample_from_uniform_distribution < self._p_exist, 1.0, 0.0)
        return logits.astype(np.float32)
    
    def _assoc_legendre_reparam_func(self, x, l, m):
        z = self._assoc_legendre_func(np.sin(0.5 * np.pi * x), l, m)
        return z

    def _assoc_legendre_func(self, x, l, m):
        z = np.zeros_like(x)

        for k in range(m, l + 1):
            c = (
                self._permutation(k, m)
                * self._combination(l, k)
                * special.binom((l + k - 1.0) / 2.0, l)
            )
            z = z + c * np.power(x, k - m)

        z = z * np.power((1.0 - np.square(x)), m / 2.0)
        return z
    
    def _gaussian(self, x, u, sigma):
        if self._use_circulant_translations:
            xdiff = self._circulant_position_diff(x, u)
        else:    
            xdiff = x - u
            
        y = np.exp(-np.square(xdiff / sigma) / 2.0)
        y = y / (np.sqrt(2 * np.pi) * sigma)
        return y
    
    def _combination(self, n, k):
        return special.factorial(n, exact=True) / (
            special.factorial(n - k, exact=True) * special.factorial(k, exact=True)
        )

    def _permutation(self, n, k):
        return special.factorial(n, exact=True) / (special.factorial(n - k, exact=True))

    def _demean(self, x, axis=1):
        return x - np.mean(x, axis=axis, keepdims=True)

    def _std_normalize(self, x, axis=1):
        return x / (np.std(x, axis=axis, keepdims=True) + self._eps)

    def _circulant_position_diff(self, x, u):
        diff = x-u
        diff_mod = np.mod(diff, self._waveform_timesteps)
        distance = np.minimum(diff_mod, self._waveform_timesteps-diff_mod)
        
        return distance
        
    @property
    def permutation_matrix(self, seed=0):
        np.random.seed(seed)
        perm = np.random.permutation(np.arange(self._waveform_timesteps))
        v = np.concatenate(
            [
                np.ones([1, 1], dtype=np.float32),
                np.zeros([1, self._waveform_timesteps - 1], dtype=np.float32),
            ],
            axis=1,
        )

        v_translated = []
        for p in perm:
            v_translated.append(np.roll(v, axis=1, shift=p))

        perm = np.concatenate(v_translated, axis=0)
        return perm

    @property
    def dst_matrix(self):
        ts_float = np.array(self._waveform_timesteps, dtype=np.float32)
        n = np.arange(1, self._waveform_timesteps + 1, dtype=np.float32)[:, None]
        k = np.arange(1, self._waveform_timesteps + 1, dtype=np.float32)[None, :]
        dst = np.sin(n * k * np.pi / (ts_float + 1.0))
        return dst / np.sqrt((self._waveform_timesteps + 1) / 2)

    @property
    def timestep_values(self):
        t = np.arange(start=0., stop=self._waveform_timesteps, dtype=np.float64)
        t = t - 0.5 * (t[0] + t[-1])
        return t
    
    @property
    def n_features(self):
        return len(self._features)

    @property
    def channels(self):
        return 1

    @property
    def n_timesteps(self):
        return self._waveform_timesteps

    @property
    def batch_size(self):
        return self._batch_size