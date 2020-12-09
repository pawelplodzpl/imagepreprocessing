from imagepreprocessing.keras_functions import (create_training_data_keras, 
make_prediction_from_array_keras, 
make_prediction_from_directory_keras, 
create_training_data_keras)


from imagepreprocessing.darknet_functions import (create_training_data_yolo, 
create_cfg_file_yolo, 
yolo_annotation_tool, 
auto_annotation_by_random_points, 
draw_bounding_boxes, 
make_prediction_from_directory_yolo,
count_classes_from_annotation_files,
remove_class_from_annotation_files)


from imagepreprocessing.utilities import train_test_split, create_confusion_matrix

source_path = "test_stuff/test_datasets/food_5class"
source_path_darknet = "test_stuff/images"

########## yolo tests ##########

create_training_data_yolo("test_stuff/test_datasets/food_5class", yolo_version=4, isTiny=True, train_machine_path_sep = "/", percent_to_use = 1.0 ,validation_split = 0.2, create_cfg_file = True)

# yolo_annotation_tool("test_stuff/images", "test_stuff/obj.names")

# auto_annotation_by_random_points("test_stuff/test_datasets/food_5class/apple_pie",1)

# draw_bounding_boxes("test_stuff\\img_pats.txt", "test_stuff\\obj.names", save_path="test_stuff\\annoted_images")

# classes = count_classes_from_annotation_files("test_stuff/images", "test_stuff/obj.names", include_zeros=True)
# print(classes)

# remove_class_from_annotation_files("test_stuff/images", 2)


########## keras and utilities tests ##########

# x, y, x_val, y_val = create_training_data_keras(source_path, save_path = None, validation_split=0.2, percent_to_use=1)

# x, y, test_x, test_y =  train_test_split(x,y, save_path=None)

# predictions = make_prediction_from_array_keras(test_x, model_path, print_output=False)

# class_names = ["elma","ayva","armut"]
# classes =     [0,0,0, 1,1,1, 2,2,2]
# predictions = [0,0,0, 0,1,1, 2,1,2]

# create_confusion_matrix(predictions,classes,class_names=class_names)






# # BUNU YAP BI ARA
# import keras
# import numpy as np
# model = keras.models.load_model(model_path)
# score = model.evaluate(test_x, test_y, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# print(score)



