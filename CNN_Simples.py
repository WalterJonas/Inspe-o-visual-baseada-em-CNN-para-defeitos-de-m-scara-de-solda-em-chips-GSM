#from sklearn.cluster import KMeans
#from sklearn import svm
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
import glob
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam


class Provir3(object):
	
	def __init__(self):

		self.path1 = '/home/jonas/Walter - ICET/Imagens Redimensionadas/'
		
		self.X_train = []
		self.Y_train = []
		self.X_test = []
		self.Y_test = []

	def loadDataset(self, path, X_train, Y_train):
		i = 0

		# Ordena as imagens a serem lidas
		self.files = [ f for f in listdir(path) if isfile(join(path,f)) ]
		self.files.sort()

		# Percorre as imagens do dataset
		print("carrega dataset....")
		for i in range( len(self.files) ):		#  i = identificador ----- 0 - qnt de imagens

			# Le cada imagem do dataset
			img = cv2.imread( join(path,self.files[i]), 1 )
			img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)    #quanto mais proximo de 0, menor a imagem
			# Armazena as imagens do dataset na lista X
			X_train.append( img )

			# Extrai o rotulo do nome da imagem e armazena na lista Y
			name = self.files[i].split("_")
			name = name[1].split(".")
			print (self.files[i].split("_"))
			if( name[0] == '0' ):
				Y_train.append( 0 )
			elif( name[0] == '1' ):
				Y_train.append( 1 )

		# Transforma as listas em arrays
		X_train = np.array(X_train)
		Y_train = np.array(Y_train)
		print("..X_train: ", X_train.shape)
		print("Y_train: ", Y_train.shape)
		
		self.lin = img.shape[0]
		self.col = img.shape[1]
		
#		X_train = X_train.reshape((X_train.shape[0], 2770, 210, 3))
#		X_train = X_train.reshape((X_train.shape[0], 554, 42, 3))
		X_train = X_train.reshape((X_train.shape[0], self.lin, self.col, 3))
		print("....X_train: ", X_train.shape)

		# Codifica os rotulos para a CNN.      ex.: (1 0 0 0 0)
		Y_train = to_categorical(Y_train)

		print("normalizacao....")
		# converte o valores do pixels de inteiros para float
		X_train = X_train.astype('float32')

		# Normaliza as imagens do array. Os valores dos pixels ficarao no intervalo entre (0 e 1).
		X_train = X_train / 255.0

		return X_train, Y_train

	# define cnn model
	def define_model(self, ):
		model = Sequential()

#		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(2770, 210, 3)))
#		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(554, 42, 3)))		
		model.add(Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_uniform', input_shape=(self.lin, self.col, 3)))		
		model.add(MaxPooling2D((2, 2)))
		
		model.add(Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_uniform'))		
		model.add(MaxPooling2D((2, 2)))
		
	#	model.add(Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_uniform'))		
	#	model.add(MaxPooling2D((2, 2)))

		model.add(Flatten())
		model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(2, activation='softmax'))
		# compile model
		opt = SGD(learning_rate=0.001, momentum=0.9)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model


	# evaluate a model using k-fold cross-validation
	def evaluate_model(self, dataX, dataY, tstX, tstY, n_folds=5):
		print("Evaluating model...")
		scores, histories = list(), list()

		# prepare cross validation
		kfold = KFold(n_folds, shuffle=True, random_state=1)
		
		c = 1
		# enumerate splits
		for train_ix, test_ix in kfold.split(dataX):
			print("fold ", c)
			# define model
			model = self.define_model()
			
			# select rows for train and test
			trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
			# fit model
			
			history = model.fit(trainX, trainY, epochs=40, batch_size=32, validation_data=(testX, testY), verbose=1)
			# evaluate model
			_, acc = model.evaluate(testX, testY, verbose=1)
			print('> %.3f' % (acc * 100.0))
			
			# stores scores
			scores.append(acc)
			histories.append(history)
			c += 1

			
#		history = model.fit(dataX, dataY, epochs=10, batch_size=32, validation_data=(tstX, tstY), verbose=1)
#		history = model.fit(dataX, dataY, epochs=10, batch_size=32, verbose=1)
		# evaluate model
		#_, acc = model.evaluate(tstX, tstY, verbose=1)
		#print('..> %.3f' % (acc * 100.0))
			
#		print("---------------------------------------------------------")
#		print("---------------------------------------------------------")
#		print("---------------------------------------------------------")
#		_, acc = model.evaluate(tstX, tstY, verbose=1)
#		print("Acuracia final: ", acc)
		
		return scores, histories

	def summarize_performance(self, scores):
		# print summary
		print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))

	# plot diagnostic learning curves
	def summarize_diagnostics(self, histories):
		for i in range(len(histories)):
			# plot loss
			plt.subplot(2, 1, 1)
			plt.title('Cross Entropy Loss')
			plt.plot(histories[i].history['loss'], color='blue', label='train')
			plt.plot(histories[i].history['val_loss'], color='orange', label='test')
			# plot accuracy
			plt.subplot(2, 1, 2)
			plt.title('Classification Accuracy')
			plt.plot(histories[i].history['accuracy'], color='blue', label='train')
			plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
		plt.show()


if __name__ == '__main__':

	obj = Provir3()

#-------------LOAD IMAGES--------------------

	print("-------------LOAD IMAGES--------------------")
	obj.X_train, obj.Y_train = obj.loadDataset(obj.path1, obj.X_train, obj.Y_train)
	print(obj.X_train.shape)
	print(obj.Y_train.shape)

#-------------CNN MODEL SETTING--------------------	
	
#	obj.define_model()
	
#-------------CNN FITTING--------------------

	scores, histories = obj.evaluate_model(obj.X_train, obj.Y_train, obj.X_test, obj.Y_test)
	
#-------------ACCURACY COMPUTING--------------------
	obj.summarize_performance(scores)
	
#-------------PLOT LOSS AND ACCURACY--------------------
	obj.summarize_diagnostics(histories)
	
	
