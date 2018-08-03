from pandas import read_csv
import pylab
from sys import argv

train_data = read_csv(argv[1])
train_step = train_data.iloc[:, 1].values
train_acc = train_data.iloc[:, 2].values

validation_data = read_csv(argv[2])
validation_step = validation_data.iloc[:, 1].values
validation_acc = validation_data.iloc[:, 2].values

pylab.plot(train_step, train_acc, 'r', label='Training')
pylab.plot(validation_step, validation_acc, 'b', label='Validation')
if argv[3] == 'Accuracy':
    pylab.legend(loc='lower right')
else:
    pylab.legend(loc='upper left')
# pylab.title('Accuracies over steps')
pylab.xlabel('Step')
pylab.ylabel(argv[3])
pylab.ylim((0.8, 1.0))
pylab.show()
