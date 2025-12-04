import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, datasets, models

# CIFAR-10 데이터셋 로드 및 전처리
(x_train, _), (_, _) = datasets.cifar10.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # -1에서 1 사이로 정규화

# 자동차(automobile) 클래스의 이미지만 선택
x_train_car = x_train[np.where((_ == 1)[0])]

# 생성자 함수 정의
def build_generator(latent_dim):
    model = models.Sequential([
        layers.Dense(128 * 8 * 8, input_dim=latent_dim),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(momentum=0.8),
        layers.ReLU(),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(momentum=0.8),
        layers.ReLU(),
        layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation='tanh')
    ])
    return model

# 판별자 함수 정의
def build_discriminator(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=3, strides=2, input_shape=input_shape, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
        layers.BatchNormalization(momentum=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.BatchNormalization(momentum=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# 생성자와 판별자 모델 생성
latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator((32, 32, 3))

# 판별자 컴파일
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# GAN 모델 생성
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = keras.Model(gan_input, gan_output)

# GAN 모델 컴파일
gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))

# GAN 모델 학습
def train_gan(epochs, batch_size=64, save_interval=50):
    for epoch in range(epochs):
        # 랜덤한 노이즈 샘플 생성
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # 가짜 이미지 생성
        generated_images = generator.predict(noise)
        
        # 진짜 이미지 샘플 선택
        idx = np.random.randint(0, x_train_car.shape[0], batch_size)
        real_images = x_train_car[idx]
        
        # 판별자 훈련
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # 생성자 훈련
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        # 훈련 과정 출력
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
        
        # 일정 간격으로 생성된 이미지 저장
        if epoch % save_interval == 0:
            save_generated_images(epoch)

# 생성된 이미지 저장 함수
def save_generated_images(epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # 이미지를 [-1, 1]에서 [0, 1] 범위로 조정
    
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.close()

# GAN 모델 학습
train_gan(epochs=2000, batch_size=64, save_interval=200)
