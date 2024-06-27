import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras import Model, layers

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

dataset = tfds.load('mnist', split='train')

batch_size = 1024
train_data = dataset.map(lambda data: tf.cast(data['image'], tf.float32) / 255.).batch(batch_size)

# Encoder 정의
class Vanila_Encoder(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim * 2)
        ])
        
    def __call__(self, x):
        # self.encoder(x)가 도출한 각각의 값을 mu, logvar로 mapping
        # tf.split: value into a list of subtensors
        mu, logvar = tf.split(self.encoder(x), 2, axis=1) 
        return mu, logvar
# Decoder 정의
class Vanila_Decoder(Model):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'), 
            layers.Dense(512, activation='relu'),
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28,28, 1))
        ])
        
    def __call__(self, z):
        return self.decoder(z)

# reparametrization
def sample(mu, logvar):
    epsilon = tf.random.normal(mu.shape)
    sigma = tf.exp(0.5 * logvar)
    return epsilon * sigma + mu

# train VAE model
def train_step(inputs):
    # GradientTape에서 gradient값들을 수집함
    with tf.GradientTape() as tape:
        # Encoder로부터 mu, logvar를 얻음 : q(z|x)
        mu, logvar = encoder(inputs) 
        # mu, logvar를 사용해서 reparameterization trick 생성
        z = sample(mu, logvar)
        # reparameterization trick을 Decoder에 넣어 reconstruct x 얻기 : (p(x|z))
        x_recon = decoder(z)
        # reconstruction loss: q(z|x)logp(x|z)
        # 입력과 생성된 이미지의 차이
        reconstruction_error = tf.reduce_sum(tf.losses.binary_crossentropy(inputs, x_recon))
        # regularization loss: KL(p(z)|q(z|x))
        # KL의 의미는?
        kl = 0.5 * tf.reduce_sum(tf.exp(logvar) + tf.square(mu) - 1. - logvar)
        # inputs.shape[0]: # of samples
        loss = (kl + reconstruction_error) / inputs.shape[0]
         # get trainable parameter
        vars_ = encoder.trainable_variables + decoder.trainable_variables 
        # get grads
        grads_ = tape.gradient(loss, vars_) 
        # apply gradient descent (update model)
        optimizer.apply_gradients(zip(grads_, vars_)) 

    return loss, reconstruction_error, kl

# Set hyperparameters
n_epochs = 50
latent_dim = 2
learning_rate = 1e-3
log_interval = 10

encoder = Vanila_Encoder(latent_dim)
decoder = Vanila_Decoder(latent_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for epoch in range(1, n_epochs + 1):    
    total_loss, total_recon, total_kl = 0, 0, 0
    for x in train_data:
        loss, recon, kl = train_step(x)
        # loss 저장
        total_loss += loss * x.shape[0]
        # error 저장
        total_recon += recon
        # total KL 저장
        total_kl += kl
    
    if epoch % log_interval == 0:
        print(
            f'{epoch:3d} iteration: ELBO {total_loss / len(dataset):.2f}, ' \
            f'Recon {total_recon / len(dataset):.2f}, ' \
            f'KL {total_kl / len(dataset):.2f}'
        )
