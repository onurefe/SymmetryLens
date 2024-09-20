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
                                                center_min, 
                                                center_max, 
                                                feature["scale_min"],
                                                feature["scale_max"],
                                                feature["amplitude_min"],
                                                feature["amplitude_max"])
            elif feature["type"] == "legendre":
                x_feature = self.lot_legendre_polynomials(num_samples,
                                                          center_min, 
                                                          center_max, 
                                                          feature["scale_min"],
                                                          feature["scale_max"],
                                                          feature["amplitude_min"],
                                                          feature["amplitude_max"],
                                                          feature["l"],
                                                          feature["m"]) 
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
    
    def _add_channels_axis(self, features):
        features = np.expand_dims(features, axis=-1)
        return features

    def _drop_channels_axis(self, features):
        return features[..., 0]

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