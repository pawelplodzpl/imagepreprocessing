## imagepreprocessing

- **Creates train ready data for keras or yolo in a single line**
- **Makes prediction process easier with using keras model from both array and directory**
- **Plots confusion matrix**


## Install

```sh
pip install imagepreprocessing
```

## Usage

```python
from imagepreprocessing import create_training_data_keras, create_training_data_yolo, make_prediction_from_directory_keras, create_confusion_matrix
```

## Create training data for keras

```python
source_path = "datasets/deep_learning/food-101/only3"
save_path = "food10class1000sampleeach"
create_training_data_keras(source_path, save_path, image_size = 299, validation_split=0.1, percent_to_use=0.1, grayscale = True, files_to_exclude=["excludemoe","hi.txt"])
```
```
File name: apple_pie - 1/3  Image:100/100
File name: baby_back_ribs - 2/3  Image:100/100
File name: baklava - 3/3  Image:100/100

validation x: 30 validation y: 30
train x: 270 train y: 270

shape of train x: (270, 299, 299, 1)
shape of train y: (270, 3)
shape of validate x: (30, 299, 299, 1)
shape of validate y: (30, 3)

file saved -> C:\Users\can\Desktop\food3class100sampleeach_x_train.pkl
file saved -> C:\Users\can\Desktop\food3class100sampleeach_y_train.pkl
file saved -> C:\Users\can\Desktop\food3class100sampleeach_x_validation.pkl
file saved -> C:\Users\can\Desktop\food3class100sampleeach_y_validation.pkl
```

## Make prediction from directory with a keras model and plot confusion matrix

```python
images_path = "deep_learning/test_images/food2"
model_path = "deep_learning/saved_models/alexnet.h5"

predictions = make_prediction_from_directory_keras(images_path, model_path)

class_names = ["apple", "melon", "orange"]
labels = [0,0,0,1,1,1,2,2,2]
create_confusion_matrix(predictions, labels, class_names=class_names)
```
```
1.jpg : 0
2.jpg : 0
3.jpg : 0
4.jpg : 1
5.jpg : 1
6.jpg : 2
7.jpg : 2
8.jpg : 2
9.jpg : 1
Confusion matrix, without normalization
[[3 0 0]
 [0 2 1]
 [0 1 2]]
```

## Make prediction and create the confusion matrix

```python
images_path = "deep_learning/test_images/food2"
save_path = "food"
model_path = "deep_learning/saved_models/alexnet.h5"

# Create training data split the data
x, y, x_val, y_val = create_training_data_keras(images_path, save_path = save_path, validation_split=0.2, percent_to_use=0.5)

# split training data
x, y, test_x, test_y =  train_test_split(x,y,save_path = save_path)

# ...
# training
# ...

class_names = ["apple", "melon", "orange"]

# make prediction
predictions = make_prediction_from_array_keras(test_x, model_path, print_output=False)

# create confusion matrix
create_confusion_matrix(predictions, test_y, class_names=class_names, one_hot=True)
```


## Make multi input model prediction and create the confusion matrix

```python
from imagepreprocessing import create_training_data_keras, train_test_split
import numpy as np

# Create training data split the data and split the data
source_path = "/content/trainingSet"
x, y = create_training_data_keras(source_path, image_size=(28,28), validation_split=0, percent_to_use=1, grayscale=True, convert_array_and_reshape=False)
x, y, test_x, test_y = train_test_split(x,y)

# prepare the data for multi input training and testing
x1 = np.array(x).reshape(-1,28,28,1)
x2 = np.array(x).reshape(-1,28,28)
y = np.array(y)
x = [x1, x2]

test_x1 = np.array(test_x).reshape(-1,28,28,1)
test_x2 = np.array(test_x).reshape(-1,28,28)
test_y = np.array(test_y)
test_x = [test_x1, test_x2]

# ...
# training
# ...

# make prediction
predictions = make_prediction_from_array_keras(test_x, "models/model.h5",print_output=False, model_summary=False, show_images=False)

# create confusion matrix
create_confusion_matrix(predictions, test_y, class_names=["0","1","2","3","4","5","6","7","8","9"], one_hot=True)

```


