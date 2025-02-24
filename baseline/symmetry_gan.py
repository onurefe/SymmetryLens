import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Layer
from synthetic_data_generator import SyntheticDataGenerator
from os.path import join

class GeneratorExpParametrized(Layer):
    def __init__(self, dims=7, **kwargs):
        super(GeneratorExpParametrized, self).__init__(**kwargs)
        self._dims = dims
        
    def build(self, input_shape):
        self._T = self.add_weight(name='generator_parametrization',
                                  shape=[self._dims, self._dims],
                                  initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1),
                                  trainable=True)
        
        super(GeneratorExpParametrized, self).build(input_shape)  # Be sure to call this at the end

    def compute_generator(self):
        A = 0.5 * (self._T - tf.transpose(self._T))
        return tf.linalg.expm(A)
    
    def call(self, X):
        G = self.compute_generator()
        return tf.matmul(X, G)
    
class GeneratorQRParametrized(Layer):
    def __init__(self, dims=7, **kwargs):
        super(GeneratorQRParametrized, self).__init__(**kwargs)
        self._dims = dims
        
    def build(self, input_shape):
        self._T = self.add_weight(name='generator_parametrization',
                                  shape=[self._dims, self._dims],
                                  initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1),
                                  trainable=True)
        
        super(GeneratorQRParametrized, self).build(input_shape)  # Be sure to call this at the end

    def compute_generator(self):
        Q, __ = tf.linalg.qr(self._T, full_matrices=True)
        return Q
    
    def call(self, X):
        G = self.compute_generator()
        return tf.matmul(X, G)
    
# define the standalone generator model
def define_generator(n_dims=7, parametrization_type="qr"):
    mymodel_inputtest = Input(shape=(n_dims,))
	
    if parametrization_type == "qr":
        mymodel_test = GeneratorQRParametrized()(mymodel_inputtest)
    elif parametrization_type == "exp":
        mymodel_test = GeneratorExpParametrized()(mymodel_inputtest)
    else:
        raise ValueError("Invalid parametrization type. Must be 'qr' or 'exp'.")
    
    model = Model(mymodel_inputtest, mymodel_test)
    return model

def define_discriminator(learning_rate = 1e-3, n_dims=7):
    model = Sequential()
    model.add(Dense(25, activation='relu', input_dim=n_dims))
    model.add(Dense(25, activation='relu'))    
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator, learning_rate=1e-3):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    
    # connect them
    model = Sequential()
	# add generator
    model.add(generator)
	# add the discriminator
    model.add(discriminator)
 
    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

def form_dataset(batch_size, dataset_size):
    dg = SyntheticDataGenerator(batch_size=batch_size, 
                                features=[{"type":"mnist_slices"}], 
                                use_circulant_translations=True,
                                waveform_timesteps=7,
                                noise_normalized_std=0.0,
                                output_representation="natural")
    num_batches = dataset_size // batch_size
    batch_datas = []
    for i in range(num_batches):
        x_batch = dg.sample_batch_of_data()
        batch_datas.append(x_batch)
    
    dataset = np.concatenate(batch_datas, axis=0)
    return dataset

def train(g_model, 
          d_model, 
          gan_model, 
          n_epochs=10000, 
          batch_size=256, 
          dataset_size=64000, 
          saving_dir="",
          log_interval_in_steps=50):
    
    num_batches = dataset_size // batch_size
    
    x = form_dataset(batch_size=batch_size, dataset_size=dataset_size)
    x = x[..., 0]
    
    for epoch in range(n_epochs):
        for b in range(num_batches):
            x_batch = x[batch_size * b: batch_size * (b+1)]
            disc_training_num_real_samples = batch_size // 4
            disc_training_num_fake_samples = batch_size // 4
            gen_training_num_real_samples = batch_size - (disc_training_num_real_samples + disc_training_num_fake_samples)
  
            x_disc_train_real = x_batch[0:disc_training_num_real_samples]
            x_disc_train_fake = x_batch[disc_training_num_real_samples:disc_training_num_real_samples + disc_training_num_fake_samples]
            x_disc_train_fake = g_model(x_disc_train_fake)		
            x_gen_train_real = x_batch[disc_training_num_real_samples + disc_training_num_fake_samples:]
    
            x_disc_train = np.concatenate([x_disc_train_real, x_disc_train_fake], axis=0)
            y_disc_train = np.concatenate([np.ones((disc_training_num_real_samples, 1)), 
                                           np.zeros((disc_training_num_fake_samples, 1))], axis=0)

            # Update discriminator
            d_loss, __ = d_model.train_on_batch(x_disc_train, y_disc_train)
  
            # Update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(x_gen_train_real, np.ones((gen_training_num_real_samples, 1)))
            
            if b % log_interval_in_steps == 0:
                print(f"Training epoch:{epoch}, step:{b}, discriminator_loss:{d_loss}, generator_loss:{g_loss}")
            
        path = join(saving_dir, f"ep{epoch}.h5")
        gan_model.save_weights(path)
