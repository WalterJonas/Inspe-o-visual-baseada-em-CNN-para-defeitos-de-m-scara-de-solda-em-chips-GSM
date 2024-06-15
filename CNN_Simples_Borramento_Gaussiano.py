import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from skimage.util import random_noise
from numpy import mean, std

class Provir3(object):
    
    def __init__(self):
        self.path1 = '/home/jonas/Walter - ICET/Imagens Redimensionadas/'
        self.X_train = []
        self.Y_train = []
        self.X_trainN = []
        self.Y_trainN = []

    def loadDataset(self, path):
        # Ordena as imagens a serem lidas
        self.files = [f for f in listdir(path) if isfile(join(path, f))]
        self.files.sort()

        # Percorre as imagens do dataset
        print("Carregando dataset...")
        for i in range(len(self.files)):
            # Le cada imagem do dataset
            img = cv2.imread(join(path, self.files[i]), 1)
            img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)  # Reduz a imagem       
        
            imgN = cv2.GaussianBlur(img, (5, 5), 0) #Pode modificar o tamanho da janela do borramento            

            # Armazena as imagens do dataset na lista X
            self.X_train.append(img)
            self.X_trainN.append(imgN)

            # Extrai o rótulo do nome da imagem e armazena na lista Y
            name = self.files[i].split("_")[1].split(".")[0]
            if name == '0':
                self.Y_train.append(0)
                self.Y_trainN.append(0)
            elif name == '1':
                self.Y_train.append(1)
                self.Y_trainN.append(1)

        # Transforma as listas em arrays
        self.X_train = np.array(self.X_train)
        self.X_trainN = np.array(self.X_trainN)
        self.Y_train = np.array(self.Y_train)
        self.Y_trainN = np.array(self.Y_trainN)
        print("..X_train: ", self.X_train.shape)
        print("Y_train: ", self.Y_train.shape)

        self.lin = img.shape[0]
        self.col = img.shape[1]

        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.lin, self.col, 3))
        self.X_trainN = self.X_trainN.reshape((self.X_trainN.shape[0], self.lin, self.col, 3))
        print("....X_train: ", self.X_train.shape)

        # Codifica os rótulos para a CNN
        self.Y_train = to_categorical(self.Y_train)
        self.Y_trainN = to_categorical(self.Y_trainN)

        print("Normalização...")
        # Converte os valores dos pixels de inteiros para float
        self.X_train = self.X_train.astype('float32')
        self.X_trainN = self.X_trainN.astype('float32')

        # Normaliza as imagens do array
        self.X_train = self.X_train / 255.0
        self.X_trainN = self.X_trainN / 255.0

    # Define cnn model
    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_uniform', input_shape=(self.lin, self.col, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(2, activation='softmax'))
        # Compile model
        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Evaluate a model using k-fold cross-validation
    def evaluate_model(self, dataX, dataY, dataXN, dataYN, n_folds=5):
        print("Evaluating model...")
        scores, histories = list(), list()
        # Prepare cross-validation
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        c = 1
        # Enumerate splits
        for train_ix, test_ix in kfold.split(dataX):
            print("fold ", c)
            # Define model
            model = self.define_model()
            # Select rows for train and test
            trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataXN[test_ix], dataYN[test_ix]
            # Fit model
            history = model.fit(trainX, trainY, epochs=40, batch_size=32, validation_data=(testX, testY), verbose=1)
            # Evaluate model
            _, acc = model.evaluate(testX, testY, verbose=1)
            print('> %.3f' % (acc * 100.0))
            # Store scores
            scores.append(acc)
            histories.append(history)
            c += 1
        return scores, histories

    def summarize_performance(self, scores):
        # Print summary
        print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))

    # Plot diagnostic learning curves
    def summarize_diagnostics(self, histories):
        for i in range(len(histories)):
            # Plot loss
            plt.subplot(2, 1, 1)
            plt.title('Cross Entropy Loss')
            plt.plot(histories[i].history['loss'], color='blue', label='train')
            plt.plot(histories[i].history['val_loss'], color='orange', label='test')
            # Plot accuracy
            plt.subplot(2, 1, 2)
            plt.title('Classification Accuracy')
            plt.plot(histories[i].history['accuracy'], color='blue', label='train')
            plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        plt.show()

if __name__ == '__main__':
    obj = Provir3()

    # Carrega as imagens
    print("-------------LOAD IMAGES--------------------")
    obj.loadDataset(obj.path1)
    print(obj.X_train.shape)
    print(obj.Y_train.shape)

    # Avalia o modelo
    scores, histories = obj.evaluate_model(obj.X_train, obj.Y_train, obj.X_trainN, obj.Y_trainN)

    # Calcula e imprime a acurácia
    obj.summarize_performance(scores)

    # Plota as curvas de perda e acurácia
    obj.summarize_diagnostics(histories)

