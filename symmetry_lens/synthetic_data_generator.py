import numpy as np
from scipy import special

class SyntheticDataGenerator:
    def __init__(
        self,
        batch_size,
        features,
        use_circulant_translations=False,
        waveform_timesteps=33,
        noise_normalized_std=0.0,
        output_representation="natural",
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
            elif feature["type"] == "dispersed_gaussian":
                x_feature = self.lot_dispersed_gaussian(num_samples,
                                                        center_min=center_min, 
                                                        center_max=center_max,
                                                        sigma_min=feature["scale_min"],
                                                        sigma_max=feature["scale_max"],
                                                        amplitude_min=feature["amplitude_min"],
                                                        amplitude_max=feature["amplitude_max"],
                                                        phase_velocity_coeffs=feature["phase_velocity_coeffs"])
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
            tdiff = self._circulant_position_diff(t, centers, self._waveform_timesteps-1)
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
            xdiff = self._circulant_position_diff(x, u, self._waveform_timesteps-1)
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

    def _circulant_position_diff(self, x, u, period):
        diff_phase = 2 * np.pi * (x - u) / period
        z = np.cos(diff_phase) + 1j * np.sin(diff_phase)
        return period * np.angle(z) / (2*np.pi)
        
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
        t = np.linspace(-0.5, 0.5, endpoint=True, num=self._waveform_timesteps)
        t = t * (self._waveform_timesteps - 1)
            
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
    
class SyntheticDataGenerator2d:
    def __init__(
        self,
        batch_size,
        features,
        latent_x_dims = 7,
        latent_y_dims = 7,
        noise_normalized_std=0.0,
        output_representation="natural",
        p_exist = 0.5,
        num_of_lots = 5,
        flatten_output = True,
        eps=1e-6,
    ):
        self._batch_size = batch_size
        self._features = features
        self._latent_x_dims = latent_x_dims
        self._latent_y_dims = latent_y_dims
        self._noise_normalized_std = noise_normalized_std
        self._output_representation = output_representation
        self._p_exist = p_exist / len(features)
        self._num_of_lots = num_of_lots
        self._flatten_output = flatten_output
        self._eps = eps
        
        assert latent_x_dims == latent_y_dims
        
    def sample_batch_of_data(self):
        x = np.zeros(shape=[self._batch_size, self._latent_x_dims, self._latent_y_dims], dtype=np.float32)
        
        for feature in self._features:
            logits = self._sample_batch_logits() 
            num_samples = self._batch_size * self._num_of_lots
            
            center_x_min = -self._latent_x_dims
            center_x_max = self._latent_x_dims
            center_y_min = -self._latent_y_dims
            center_y_max = self._latent_y_dims
            
            if feature["type"] == "gaussian":
                x_feature = self.lot_gaussians2d(num_samples,
                                                 center_x_min=center_x_min, 
                                                 center_x_max=center_x_max,
                                                 center_y_min=center_y_min,
                                                 center_y_max=center_y_max,
                                                 sigma_x_min=feature["scale_x_min"],
                                                 sigma_x_max=feature["scale_x_max"],
                                                 sigma_y_min=feature["scale_y_min"],
                                                 sigma_y_max=feature["scale_y_max"],
                                                 amplitude_min=feature["amplitude_min"],
                                                 amplitude_max=feature["amplitude_max"])
            else:
                raise ValueError('Feature type is not defined.')
                
            x_feature = np.reshape(x_feature, newshape=[self._batch_size, 
                                                        self._num_of_lots, 
                                                        self._latent_x_dims, 
                                                        self._latent_y_dims])
            
            x_feature = np.sum(x_feature * (logits[:, :, np.newaxis, np.newaxis]), axis=1)
            x = x + x_feature
        
        x = self._std_normalize(x, axis=(0, 1, 2))                
        x = self._add_noise(x)
        
        if self._flatten_output:
            x = np.reshape(x,
                           newshape=[self._batch_size, self._latent_x_dims * self._latent_y_dims])
            
        x = np.expand_dims(x, axis=-1)

        if self._output_representation == "natural":
            x_final = x
        elif self._output_representation == "permuted":
            x_final = self._apply_permutation_map(x)
        else:
            raise ValueError('Only natural and permuted representation types are supported for 2d.')
        
        return x_final
        
    def lot_gaussians2d(self, 
                        num_samples, 
                        center_x_min, 
                        center_x_max, 
                        center_y_min, 
                        center_y_max, 
                        sigma_x_min, 
                        sigma_x_max,
                        sigma_y_min,
                        sigma_y_max, 
                        amplitude_min, 
                        amplitude_max):
        x_centers = np.random.uniform(center_x_min, center_x_max, size=[num_samples, 1])
        y_centers = np.random.uniform(center_y_min, center_y_max, size=[num_samples, 1])
        x_sigmas = np.random.uniform(sigma_x_min, sigma_x_max, size=[num_samples, 1])
        y_sigmas = np.random.uniform(sigma_y_min, sigma_y_max, size=[num_samples, 1])
        amplitudes = np.random.uniform(amplitude_min, amplitude_max, size=[num_samples])
        centers = np.concatenate([x_centers, y_centers], axis=1)
        sigmas = np.concatenate([x_sigmas, y_sigmas], axis=1)

        # Expand dimensions to make them broadcastable
        centers = centers[:, np.newaxis, np.newaxis, :]    # Shape: [num_samples, 1, 1, 2]
        sigmas = sigmas[:, np.newaxis, np.newaxis, :]      # Shape: [num_samples, 1, 1, 2]
        amplitudes = amplitudes[:, np.newaxis, np.newaxis] # Shape: [num_samples, 1, 1]

        # Create a grid of coordinates
        t_x = np.linspace(-0.5, 0.5, endpoint=True, num=self._latent_x_dims)
        t_x = t_x * (self._latent_x_dims - 1)
        t_y = np.linspace(-0.5, 0.5, endpoint=True, num=self._latent_y_dims)
        t_y = t_y * (self._latent_y_dims - 1)
        t_xx, t_yy = np.meshgrid(t_x, t_y, indexing='ij')
        t_grid = np.stack([t_xx, t_yy], axis=-1)           # Shape: [latent_x_dims, latent_y_dims, 2]

        # Expand grid to match the number of samples
        t = t_grid[np.newaxis, :, :, :]                    # Shape: [1, latent_x_dims, latent_y_dims, 2]
        t = np.repeat(t, num_samples, axis=0)              # Shape: [num_samples, latent_x_dims, latent_y_dims, 2]

        # Compute the 2D Gaussian values
        x = amplitudes * self._gaussian2d(t, centers, sigmas)  # Shape: [num_samples, latent_x_dims, latent_y_dims]
        
        x = np.reshape(x, newshape=[num_samples, self._latent_x_dims * self._latent_y_dims])
        return x
    
    def _add_noise(self, x_batch):
        x_batch_std = np.std(x_batch, dtype=np.float32, keepdims=True)
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
    
    def _gaussian2d(self, x, u, sigma):
        xdiff = x - u                                      # Shape: [num_samples, waveform_timesteps, waveform_timesteps, 2]

        xdiff_norm = xdiff / sigma                             # Normalize differences
        xdiff_norm_sq = np.square(xdiff_norm)                  # Square the normalized differences
        xdiff_norm_sq_sum = np.sum(xdiff_norm_sq, axis=-1)     # Sum over the last axis (x and y)

        # Compute the 2D Gaussian
        y = np.exp(-xdiff_norm_sq_sum / 2.0)
        y = y / (2 * np.pi * sigma[..., 0] * sigma[..., 1])    # Normalize the Gaussian
        return y

    def _apply_permutation_map(self, x):
        x = np.transpose(x, axes=[0, 2, 1])
        x = np.matmul(x, self.permutation_matrix)
        x = np.transpose(x, axes=[0, 2, 1])
        return x
    
    def _demean(self, x, axis=1):
        return x - np.mean(x, axis=axis, keepdims=True)

    def _std_normalize(self, x, axis=1):
        return x / (np.std(x, axis=axis, keepdims=True) + self._eps)
    
    @property
    def permutation_matrix(self, seed=0):
        np.random.seed(seed)
        out_size = self._latent_x_dims * self._latent_y_dims
        perm = np.random.permutation(np.arange(out_size))
        v = np.concatenate(
            [
                np.ones([1, 1], dtype=np.float32),
                np.zeros([1, out_size - 1], dtype=np.float32),
            ],
            axis=1,
        )

        v_translated = []
        for p in perm:
            v_translated.append(np.roll(v, axis=1, shift=p))

        perm = np.concatenate(v_translated, axis=0)
        return perm
    
    @property
    def latent_y_dims(self):
        return self._latent_y_dims
    
    @property
    def latent_x_dims(self):
        return self._latent_x_dims
    
    @property
    def n_dims(self):
        return self._latent_x_dims * self._latent_y_dims
    
    @property
    def channels(self):
        return 1

    @property
    def batch_size(self):
        return self._batch_size