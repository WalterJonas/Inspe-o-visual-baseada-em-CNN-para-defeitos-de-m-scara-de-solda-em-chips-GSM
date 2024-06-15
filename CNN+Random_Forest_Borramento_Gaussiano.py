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
from sklearn.ensemble import RandomForestClassifier
from numpy import mean, std
from skimage.util import random_noise

class Provir3(object):
    
    def __init__(self):
        self.path1 = '/home/jonas/Walter - ICET/Imagens Redimensionadas/'
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        
        self.X_trainN = []
        self.Y_trainN = []
        self.X_testN = []
        self.Y_testN = []

    def loadDataset(self, path, X_train, Y_train, X_trainN, Y_trainN):
        self.files = [f for f in listdir(path) if isfile(join(path, f))]
        self.files.sort()

        print("Carregando dataset...")
        for i in range(len(self.files)):
            img = cv2.imread(join(path, self.files[i]), 1)
            img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
            
            imgN = cv2.GaussianBlur(img, (3, 3), 0) #Pode modificar o tamanho da janela do borramento 
            
            X_train.append(img)
            X_trainN.append(imgN)
           
            name = self.files[i].split("_")
            name = name[1].split(".")
            print (self.files[i].split("_"))
            if name[0] == '0':
                Y_train.append(0)
                Y_trainN.append(0)
            elif name[0] == '1':
                Y_train.append(1)
                Y_trainN.append(1)

        X_train = np.array(X_train)
        X_trainN = np.array(X_trainN)
        Y_train = np.array(Y_train)
        Y_trainN = np.array(Y_trainN)

        self.lin = img.shape[0]
        self.col = img.shape[1]

        X_train = X_train.reshape((X_train.shape[0], self.lin, self.col, 3))
        X_trainN = X_trainN.reshape((X_trainN.shape[0], self.lin, self.col, 3))

        Y_train = to_categorical(Y_train)
        Y_trainN = to_categorical(Y_trainN)

        X_train = X_train.astype('float32')
        X_trainN = X_trainN.astype('float32')
        
        X_train = X_train / 255.0
        X_trainN = X_trainN / 255.0

        return X_train, Y_train, X_trainN, Y_trainN

    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_uniform', input_shape=(self.lin, self.col, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(2, activation='softmax'))
        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def evaluate_model(self, dataX, dataY, tstX, tstY, dataXN, dataYN, tstXN, tstYN, n_folds=5):
        print("Avaliando modelo...")
        scores, histories, rf_accuracies = [], [], []
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        for train_ix, test_ix in kfold.split(dataX):
            if len(train_ix) == 0 or len(test_ix) == 0:
                print("Conjunto de treinamento ou teste vazio. Pulando esta divisão.")
                continue
            model = self.define_model()
            trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataXN[test_ix], dataYN[test_ix]
            history = model.fit(trainX, trainY, epochs=40, batch_size=32, validation_data=(testX, testY), verbose=1)
            _, acc = model.evaluate(testX, testY, verbose=1)
            print('> CNN Accuracy: %.3f' % (acc * 100.0))
            scores.append(acc)
            histories.append(history)

            # Extração de características das camadas
            feature_extractor = Sequential(model.layers[:-2])
            features_train = feature_extractor.predict(trainX)
            features_test = feature_extractor.predict(testX)

            # Treinamento e avaliação do Random Forest
            rf_classifier = RandomForestClassifier()
            rf_classifier.fit(features_train, np.argmax(trainY, axis=1))
            rf_accuracy = rf_classifier.score(features_test, np.argmax(testY, axis=1))
            print('> Random Forest Accuracy: %.3f' % (rf_accuracy * 100.0))
            rf_accuracies.append(rf_accuracy)

        return scores, histories, rf_accuracies

    def summarize_performance(self, scores, rf_accuracies):
        print('CNN Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores) * 100, np.std(scores) * 100, len(scores)))
        print('Random Forest Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(rf_accuracies) * 100, np.std(rf_accuracies) * 100, len(rf_accuracies)))

    def summarize_diagnostics(self, histories):
        for i in range(len(histories)):
            plt.subplot(2, 1, 1)
            plt.title('Cross Entropy Loss')
            plt.plot(histories[i].history['loss'], color='blue', label='train')
            plt.plot(histories[i].history['val_loss'], color='orange', label='test')
            plt.subplot(2, 1, 2)
            plt.title('Classification Accuracy')
            plt.plot(histories[i].history['accuracy'], color='blue', label='train')
            plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        plt.show()

if __name__ == '__main__':
    obj = Provir3()

    #-------------LOAD IMAGES--------------------
    print("-------------LOAD IMAGES--------------------")
    obj.X_train, obj.Y_train, obj.X_trainN, obj.Y_trainN = obj.loadDataset(obj.path1, obj.X_train, obj.Y_train, obj.X_trainN, obj.Y_trainN)
    print(obj.X_train.shape)
    print(obj.Y_train.shape)

    #-------------CNN MODEL SETTING--------------------
    # obj.define_model()

    #-------------CNN FITTING--------------------
    scores, histories, rf_accuracies = obj.evaluate_model(obj.X_train, obj.Y_train, obj.X_test, obj.Y_test, obj.X_trainN, obj.Y_trainN, obj.X_testN, obj.Y_testN)

    #-------------ACCURACY COMPUTING--------------------
    obj.summarize_performance(scores, rf_accuracies)

    #-------------PLOT LOSS AND ACCURACY--------------------
    #obj.summarize_diagnostics(histories)

