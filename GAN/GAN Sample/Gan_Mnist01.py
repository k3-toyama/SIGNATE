import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Convolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import np_utils
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from tqdm import tqdm

class Gan:
    def __init__(self,image_shape,input_size_of_generator,IsCnn=False):
        self.img_shape=image_shape
        self.input_size_of_generator=input_size_of_generator

        self.optimizer=Adam(0.0002, 0.5)
        self.generator=self.create_generator()
        self.discriminator=self.create_discriminator()

        if IsCnn:
            self.generator=self.create_cnn_generator()
            self.discriminator=self.create_cnn_discriminator()

        self.discriminator.compile(loss="binary_crossentropy",
                                   optimizer=self.optimizer,
                                   metrics=["accuracy"])

        self.combined_model=self.combine_for_generator()
        self.combined_model.compile(loss="binary_crossentropy",
                                   optimizer=self.optimizer)








    def create_generator(self):
        """
        Generate a Generetor model.
        :return:Generetor model
        """
        noize_shape=(self.input_size_of_generator,)
        img_shape=np.prod(self.img_shape)

        model=Sequential()

        model.add(Dense(256,input_shape=noize_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(img_shape, activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()

        return model


    def create_discriminator(self):
        """
        Generate a discriminator model.
        :return:discriminator model
        """
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        return model

    def create_cnn_generator(self):
        """
        Generate a Generetor model by CNN.
        :return:
        """
        model=Sequential()

        model.add(Dense(1024,input_shape=(self.input_size_of_generator,)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128 * 7 * 7))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
        model.add(UpSampling2D((2, 2)))
        model.add(Convolution2D(64, 5, 5, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((2, 2)))
        model.add(Convolution2D(1, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        model.summary()
        return model

    def create_cnn_discriminator(self):
        """
        Generate a Discriminator model by CNN.
        :return:
        """
        model = Sequential()
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2),
                                border_mode='same', input_shape=self.img_shape))
        model.add(LeakyReLU(0.2))
        model.add(Convolution2D(128, 5, 5, subsample=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()

        return model

    def combine_for_generator(self):
        """
        Combine Generator model and Discriminator model for training Generator model.
        Clipping the varues of Discriminator model.
        :return:combined model
        """
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model

    def combine_for_discriminator(self):
        """
        Combine the both of the models for training Discriminator model.
        Generator model generate and pass the images to Discriminator model , and Discriminator model
        classification the images are True or False.
        :return:combined model
        """
        z = Input(shape=(self.input_size_of_generator,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        model = Model(z, valid)
        model.summary()
        return model

    def save_img(self,epoch,intr,page=9,path="./picture"):
        """
        Generate image by Generator model ,and save image.
        :return:
        """
        """
      
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        np.random.seed(1122345)
        noize = np.random.normal(0, 1, (page, self.input_size_of_generator))
        np.random.seed()
        img = np.array(self.generator.predict(noize))
        img = np.reshape(img, [-1, self.img_shape[0], self.img_shape[1]])
        img = 127.5 * img + 127.5

        interval = int(np.sqrt(page))
        fig = plt.figure()
        for i in range(page):
            axs = fig.add_subplot(interval, interval, i + 1)
            axs.imshow(img[i], "gray")

        plt.savefig(path + '/figure_' + str(epoch)+"_"+str(intr) + '.png')
        plt.close()

    def save_model(self,path):
        self.generator.save(path+"/model/model_gan.h5")




    def train(self,epochs,data,path,batch=128,save_interval=100):
        x_train=(data.astype(np.float32)-np.max(data))/np.max(data)

        half_batch=int(batch/2)
        Batch_size=int(x_train.shape[0] / half_batch)

        interval=0
        for epoch in range(epochs):

            for iteration in tqdm(range(Batch_size)):
                # ---------------------
                # Train Discriminator model
                # ---------------------

                # Generate half batch data by Generator model
                noize = np.random.normal(0, 1, (half_batch, self.input_size_of_generator))
                gen_img = self.generator.predict(noize)

                # Pick up half batch data in true data
                idx = np.random.randint(0, x_train.shape[0], half_batch)
                re_img = X_train[idx]
                # re_img=np.random.choice(x_train,half_batch)

                # Train Discriminator model
                d_loss_real = self.discriminator.train_on_batch(re_img, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_img, np.zeros((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)  # mean of all loss

                # ---------------------
                # Train Generator model
                # ---------------------

                # Recreate noize and clipping label to one
                noize = np.random.normal(0, 1, (batch, self.input_size_of_generator))
                label = np.ones((batch, 1))
                # Train Generator
                g_loss = self.combined_model.train_on_batch(noize, label)
                if epoch % save_interval==0 and interval%500==0:
                    print("Now Epoch is ", epoch, "interval = ",interval," D_loss =", d_loss[0], " acc= ", d_loss[1], " G_loss = ", g_loss)
                    self.save_img(epoch,interval,path=path)

                interval+=1
            self.save_model(path)












if __name__=="__main__":
    save_path="../picture/DCGAN"

    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    shape=[28,28,1]
    gan=Gan(shape,100,IsCnn=True)

    Epoch=500
    batch=32
    data=X_train

    gan.train(Epoch,data,save_path,batch=batch,save_interval=1)


























