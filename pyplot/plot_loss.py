from pandas import read_csv
import pylab
from sys import argv

train_data_8s = read_csv('FCN-8s-ResNet101_Vaihingen/run_train-tag-entropy_1.csv')
train_step_8s = train_data_8s.iloc[:, 1].values
train_acc_8s = train_data_8s.iloc[:, 2].values

validation_data_8s = read_csv('FCN-8s-ResNet101_Vaihingen/run_validation-tag-entropy_1.csv')
validation_step_8s = validation_data_8s.iloc[:, 1].values
validation_acc_8s = validation_data_8s.iloc[:, 2].values

pylab.plot(train_step_8s, train_acc_8s, 'green', label='Training with 2 skips')
pylab.plot(validation_step_8s, validation_acc_8s, 'purple', label='Validation 2 skips')

train_data_4s = read_csv('FCN-4s-ResNet101_Vaihingen/run_train-tag-entropy_1.csv')
train_step_4s = train_data_4s.iloc[:, 1].values
train_acc_4s = train_data_4s.iloc[:, 2].values

validation_data_4s = read_csv('FCN-4s-ResNet101_Vaihingen/run_validation-tag-entropy_1.csv')
validation_step_4s = validation_data_4s.iloc[:, 1].values
validation_acc_4s = validation_data_4s.iloc[:, 2].values

pylab.plot(train_step_4s, train_acc_4s, 'r', label='Training with 3 skips')
pylab.plot(validation_step_4s, validation_acc_4s, 'b', label='Validation 3 skips')

pylab.legend(loc='upper left')
pylab.xlabel('Step')
pylab.ylabel('Loss')
pylab.show()
