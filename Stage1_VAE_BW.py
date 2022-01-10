## Setup ##
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid the error "CPU supports"
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' # Avoid the error "Not creating XLA devices, tf_xla_enable_xla_devices not set"



## Make a dataset with shuffling ##
path_dir = "C:/Users/mirac/Documents/Pycharm/VAE/EIT_FER/"
file_list = os.listdir(path_dir)
data_num = len(file_list)
temp_stacking = []
count = 0

for img in file_list:
    np_img = np.asarray(Image.open(path_dir + img)) / 255
    # tensor_img = tf.convert_to_tensor(np_img)
    # temp_stacking.append(tensor_img)
    temp_stacking.append(np_img)
    count += 1
    print(str(count) + " / " + str(data_num) + " Stack Finished ...")

shuffled_img = np.random.permutation(temp_stacking)
shuffled_img = np.expand_dims(shuffled_img, -1)
print("Pixel value normalize & shuffling Finished ...")



# ## Create a sampling layer ## NOT Use
# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
#
#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon



## Build the encoder ##

encoder_path = "C:/Users/mirac/Documents/Pycharm/VAE/" + 'encoder_EIT_FER.h5'
if (os.path.exists(encoder_path)):
    encoder = keras.models.load_model(encoder_path, compile=False)
    encoder.summary()
    print("Decoder model exist & loaded ...")

else:
    # Original : 409 X 680 X 3
    # Modified : 128 X 128 X 3
    latent_dim = 16
    encoder_inputs = keras.Input(shape=(128, 128, 1))
    x = layers.Conv2D(8, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()



## Build the decoder ##
decoder_path = "C:/Users/mirac/Documents/Pycharm/VAE/" + 'decoder_EIT_FER.h5'
if (os.path.exists(decoder_path)):
    decoder = keras.models.load_model(decoder_path, compile=False)
    decoder.summary()
    print("Decoder model exist & loaded ...")

else:
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((8, 8, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(8, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 8, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()



## Define the VAE as a `Model` with a custom 'train_step' ##
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }



## Train the VAE ##
# (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
# mnist_digits = np.concatenate([x_train, x_test], axis=0)
# shuffled_img = np.expand_dims(shuffled_img, -1).astype("float32") / 255

x_train, dummy = train_test_split(shuffled_img, test_size=0.5)
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(x_train, epochs=300, batch_size=10)

z_sample = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
x_decoded = vae.decoder.predict(z_sample)
plt.imshow(np.reshape(x_decoded, (128, 128, 1)))
plt.axis('off')
plt.show()

## Latent vector layer code ##
# loc_z = len(encoder.layers) - 1
# print(encoder.layers[loc_z].output_shape)

# Save encoder
encoder.save(encoder_path)
print("Encoder model saved ... ")
# Save decoder
decoder.save(decoder_path)
print("Decoder model saved ... ")