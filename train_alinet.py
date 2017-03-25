# coding: utf-8

import sys
import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from def_alinet import create_alinet
from keras import metrics

def train_network(augment_data=True, nb_epochs=15000, batch_size=256, loss='categorical_crossentropy', optim = 'adam', X_train_file = "X_train_FERFalse_48_48_3.npy", X_test_file = "X_test_FERFalse_48_48_3.npy", Y_train_file = "Y_train_FERFalse_48_48_3.npy", Y_test_file = "Y_test_FERFalse_48_48_3.npy", model_from = None, logger=True, lr_reduce=True, min_lr = 0.0001, metrics = ['accuracy']):

	X_train = np.load(X_train_file)
	X_test = np.load(X_test_file)
	Y_train = np.load(Y_train_file)
	Y_test = np.load(Y_test_file)

#------------------------dataaugmentation---------------------------------------------------#

	if (augment_data):

		datagen = ImageDataGenerator(
			featurewise_center=True,
			featurewise_std_normalization=True,
			rotation_range=20,
			width_shift_range=0.2,
			height_shift_range=0.2,
			horizontal_flip=True)

		# compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied)
		datagen.fit(X_train)

		print "X_training_shape: ",X_train.shape
		print "X_testing_shape: ",X_test.shape
		print "Y_training_shape: ",Y_train.shape
		print "training_shape: ",Y_test.shape
	if model_from:
		model = model_from
	else:
		model = create_alinet(input_shape = (3, 48, 48))
	#nb_epochs = int(argv[0])
	if logger:
		csv_logger = CSVLogger('training_from_scratch_'+'alinet'+str(nb_epochs)+'.log', separator=',', append=False)
	if lr_reduce:
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=min_lr)

	callbacks = []
	if logger:
		callbacks.append(csv_logger)
	if lr_reduce:
		callbacks.append(reduce_lr)
#-----------------------------------model compilation------------------------------------#

	model.compile(optimizer=optim, loss=loss, metrics=metrics)

#-----------------------------Actual Training-------------------------------------------#
	if (augment_data):
		print "\nEpochs argument: ",(int(nb_epochs))

		model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), callbacks=callbacks, nb_epoch= int(nb_epochs), samples_per_epoch=len(X_train), validation_data=(X_test, Y_test))
	else:
		model.fit(X_train, Y_train, callbacks = callbacks , batch_size=batch_size, nb_epoch = nb_epochs, verbose=1, validation_data=(X_test, Y_test))

	loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
	metrics_file = open('alinet_CKplus_data_augmented_metrics.txt', 'a')
	metrics_file.write("\nloss: "+str(float(loss))+"\n")
	metrics_file.write("accuracy: "+str(float(accuracy)))
	metrics_file.write("\nOptimizer, epochs: sgd_initlr=0.01, "+str(nb_epochs))
	metrics_file.close()

	# serialize weights to HDF5
	model.save_weights("alinet_model_CKplus_" + str(nb_epochs) + "_epochs.h5")
	print("Saved weights to disk\n")
