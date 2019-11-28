import os
import random
import pickle
import itertools 
from shutil import copyfile


def __write_to_file(to_write, file_name):
    with open(file_name,'w', encoding='utf-8') as file:
        for item in to_write:
            file.write(item.__str__())
            file.write("\n")
    


def create_training_data_keras(source_path, save_path = None, img_size = 224, percent_to_use = 1, validation_split = 0, grayscale = False, one_hot = True, shuffle = True, numpy_array = True, files_to_exclude = [".DS_Store",""]):
    """
    Creates train ready data for classification from image data
    Takes all the image directories alphabetically in a main directory 

    # Arguments:
        source_path: source path of the images see input format
        save_path (None): save path for clean training data 
        img_size (224): size of the images for resizing
        percent_to_use (1): percentage of data that will be used
        validation_split (0.2): splits validation data with given percentage give 0 if you don't want validation split
        grayscale (False): converts images to grayscale
        one_hot (True): makes one hot encoded y train if True if not uses class indexes as labels
        shuffle (True): shuffle the data
        numpy_array (True): converts list to numpy array if True
        files_to_exclude ([".DS_Store",""]): list of file names to exclude in the image directory (can be hidden files)

    # Returns:
        List or numpy array of train data optionally validation data

    # Save:
        Saves x train and y train optionally validation x and y 
        Save format is .pkl (pickle data)
        If you want you can prevent saveing the file by sending None as save_path

    # Input format:
        source_path = some_dir
        
        /some_dir
        ├──/class1
            ├──img1.jpg
            ├──img2.jpg
        ├──/class2
            ├──img1.jpg

    # output format:
        save_path = save/food_data

        save/food_data_x_train.pkl
        save/food_data_y_train.pkl   
        save/food_data_x_validation.pkl
        save/food_data_y_validation.pkl   
        
    # Example:
        ``python
            source_path = "C:\\Users\\can\\datasets\\deep_learning\\food-101\\only3"
            save_path = "C:\\Users\\can\\Desktop\\food10class1000sampleeach"
            create_training_data_keras(source_path, save_path, img_size = 299, validation_split=0.1, percent_to_use=0.1, grayscale = True, files_to_exclude=["excludemoe","hi.txt"])
        ``                      
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    x = []
    y = [] 
    x_val = []
    y_val = []

    CATEGORIES = os.listdir(source_path)  # get all file names from main dir
    CATEGORIES.sort()                     # sort the directories

    # remove excluded files
    for exclude in files_to_exclude:
        if exclude in CATEGORIES: 
            CATEGORIES.remove(exclude)
    

    # loop in the main directory
    for category_index, category in enumerate(CATEGORIES):

        path = os.path.join(source_path, category)
        number_of_categories = len(CATEGORIES)
        index_of_category = CATEGORIES.index(category)
        images = os.listdir(path)


        # fix possible percentage errors
        if(validation_split < 0 or validation_split > 1):
            print("Enter a possible validation_split between 0 and 1")
            return

        if(percent_to_use <= 0 or percent_to_use > 1):
            print("Enter a possible percentage between 0 and 1")
            return
        elif(int(percent_to_use * len(images)) == 0):
            print("Percentage is too small for this set")
            return
        else:
            stop_index = int(len(images)*percent_to_use)


        # loop inside each category folder with itertools for stoping on a percentage
        for image_index, img in enumerate(itertools.islice(images , 0, stop_index)):

            # print percent info
            print("File name: {} - {}/{}  Image:{}/{}".format(category, index_of_category+1, number_of_categories, image_index+1, stop_index), end="\r")
            
            # there can be broken images
            try:
                # convert grayscale
                if(grayscale):
                    temp_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                else:
                    temp_array = cv2.imread(os.path.join(path, img)) 

                # resize image
                img_array = cv2.resize(temp_array, (img_size, img_size))   

                # one hot encoding
                if(one_hot):  
                    temp_y = []
                    for i in range(len(CATEGORIES)):
                        if(i == category_index):
                            temp_y.append(1)
                        else:
                            temp_y.append(0)
                    y.append(temp_y)
                # if one hot is not selected use index of the file as label
                else:
                    y.append(index_of_category)  

                x.append(img_array)
            except:
                pass

        print("")


    if(shuffle):
        xy = list(zip(x,y))
        random.shuffle(xy)
        x, y = list(zip(*xy))
    

    # validation split
    if(validation_split != 0):
        if(int(validation_split * len(images)) == 0):
            print("Validation split is too small for this set")
            return

        # split
        train_percent = int(len(x) - (validation_split * len(x)))
        x_val = x[train_percent:]
        y_val = y[train_percent:]
        x = x[:train_percent]
        y = y[:train_percent]

        print("\nvalidation x: {0} validation y: {1}".format(len(x_val),len(y_val)))

    print("train x: {0} train y: {1}\n".format(len(x),len(y)))

    # array conversion
    if(numpy_array):
        if(grayscale):
            third_dimension = 1
        else:
            third_dimension = 3
    
        x = np.array(x).reshape(-1, img_size, img_size, third_dimension)
        y = np.array(y)

        print("shape of train x: {0}\nshape of train y: {1}".format(x.shape,y.shape))

        if(validation_split != 0):
            x_val = np.array(x_val).reshape(-1, img_size, img_size, third_dimension)
            y_val = np.array(y_val)    
            print("shape of validate x: {0}\nshape of validate y: {1}".format(x_val.shape,y_val.shape))
    

    # save
    if(save_path != None):
        with open(save_path + "_x_train.pkl", "wb") as file:
            pickle.dump(x, file, protocol=pickle.HIGHEST_PROTOCOL)
            print("\nfile saved -> {0}{1}".format(save_path,"_x_train.pkl"))

        with open(save_path + "_y_train.pkl", "wb") as file:
            pickle.dump(y, file, protocol=pickle.HIGHEST_PROTOCOL)
            print("file saved -> {0}{1}".format(save_path,"_y_train.pkl"))
        
        if(validation_split != 0):
            with open(save_path + "_x_validate.pkl", "wb") as file:
                pickle.dump(x_val, file, protocol=pickle.HIGHEST_PROTOCOL)
                print("file saved -> {0}{1}".format(save_path,"_x_validation.pkl"))

            with open(save_path + "_y_validate.pkl", "wb") as file:
                pickle.dump(y_val, file, protocol=pickle.HIGHEST_PROTOCOL)
                print("file saved -> {0}{1}\n".format(save_path,"_y_validation.pkl"))
        

    return x, y, x_val, y_val



def create_training_data_yolo(source_path, save_file_name = "obj", percent_to_use = 1, validation_split = 0.2, files_to_exclude = [".DS_Store","train.txt","test.txt"]):
    """
    Creates train ready data for yolo labels all the images by center automatically
    (This is not the optimal way of labeling but if you need a lot of data fast this is an option)

    # Arguments:
        source_path: source path of the images see input format
        save_file_name (obj): save path name to copy images and to create txt files with labels
        percent_to_use (1): percentage of data that will be used
        validation_split (0.2): splits validation data with given percentage give 0 if you don't want validation split
        files_to_exclude ([".DS_Store","train.txt","test.txt"]): list of file names to exclude in the image directory (can be hidden files)

    # Save:
        Copies all images in to save_file_name directory and creates txt files for each image see output format

    # Input format:
        source_path = some_dir
        
        /some_dir
        ├──/class1
            ├──img1.jpg
            ├──img2.jpg
        ├──/class2
            ├──img3.jpg

    # output format:
        source_path = some_dir
        save_file_name = "obj"
        
        /some_dir
        ├──/obj
            ├──img1.jpg
            ├──img1.txt
            ├──img2.jpg
            ├──img2.txt
            ├──img3.jpg
            ├──img3.txt

    # Example:
        ``python
            source_path = "food-101\\images"
            create_training_data_yolo(source_path)       
         ``                      
    """

    image_names_train = []
    image_names_test = []

    CATEGORIES = os.listdir(source_path)  # get all file names from main dir
    CATEGORIES.sort()                     # sort the directories

    # remove excluded files
    for exclude in files_to_exclude:
        if exclude in CATEGORIES: 
            CATEGORIES.remove(exclude)
    
    # make the dir
    if not os.path.exists(os.path.join(source_path, save_file_name)):
        os.makedirs(os.path.join(source_path, save_file_name))
    else:
        CATEGORIES.remove(save_file_name)

    # loop in the main directory
    for category_index, category in enumerate(CATEGORIES):


        path = os.path.join(source_path, category)
        number_of_categories = len(CATEGORIES)
        index_of_category = CATEGORIES.index(category)
        images = os.listdir(path)

        # fix possible percentage error
        if(percent_to_use <= 0 or percent_to_use > 1):
            print("Enter a possible percentage between 0 and 1")
            return
        elif(int(percent_to_use * len(images)) == 0):
            print("Percentage is too small for this set")
            return
        else:
            stop_index = int(len(images)*percent_to_use)

        image_names = []       

        # loop inside each category folder   itertools for stoping on a percentage
        for image_index, img in enumerate(itertools.islice(images , 0, stop_index)):

            # percent info
            print("File name: {} - {}/{}  Image:{}/{}".format(category, index_of_category+1, number_of_categories, image_index+1, stop_index), end="\r")

        
            # yolo label format
            # <object-class> <x_center> <y_center> <width> <height>
            # class 0.5 0.5 1 1 

            yolo_labels = "{} {} {} {} {}".format(category_index, 0.5, 0.5, 1, 1)
            
            save_path = os.path.join(source_path, save_file_name)

            basename, _ = os.path.splitext(img)
            text_name = basename + ".txt"
            path_for_txt_file = os.path.join(save_path, text_name)
 
            __write_to_file([yolo_labels], path_for_txt_file)

            # copy_files_to_new_path
            new_path_img = os.path.join(save_path, img)            
            copyfile(os.path.join(path, img), new_path_img)

            image_names.append("data/obj/" + img)
        
        print("")

        train_percent = int(len(images) - (validation_split * len(images)))
        
        image_names_train += image_names[:train_percent]
        image_names_test += image_names[train_percent:]


    __write_to_file(image_names_train, file_name = os.path.join(source_path, "train.txt"))
    __write_to_file(image_names_test, file_name = os.path.join(source_path, "test.txt"))



def make_prediction(images_path, keras_model_path, image_size = 224, model_summary=True, show_images=False, grayscale = False, files_to_exclude = [".DS_Store",""]):
    """
    Reads test data from directory resizes it and makes prediction with using a keras model

    # Arguments:
        images_path: source path of the test images see input format
        keras_model_path: path of the keras model 
        img_size (224): size of the images for resizing
        model_summary (True): shows keras model summary 
        show_images (False): shows the predicted image
        grayscale (False): converts images to grayscale
        files_to_exclude ([".DS_Store",""]): list of file names to exclude in the image directory (can be hidden files)

    # Returns:
        Prediction results in a list
    
    # Input format:
        images_path = some_dir
        
        /some_dir
            ├──img1.jpg
            ├──img2.jpg

    # Example:
        ``python
            images_path = "test_images\\food2"
            model_path = "saved_models\\a.h5"
            make_prediction(images_path, model_path, show_images=True)
        ``
    """

    import warnings
    warnings.filterwarnings("ignore")

    import matplotlib.pyplot as plt
    import numpy as np
    import keras
    import cv2

    test_images = []
    test_image_names = []

    images = os.listdir(images_path)
    images.sort()

    # remove excluded files
    for exclude in files_to_exclude:
        if exclude in images: 
            images.remove(exclude)

    # load model
    model = keras.models.load_model(keras_model_path)

    # get all images
    for image in images:
        abs_path = os.path.join(images_path, image)

        try:
            if(grayscale):
                third_dimension = 1
                img_array = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
            else:
                third_dimension = 3
                img_array = cv2.imread(abs_path)

            new_array = cv2.resize(img_array, (image_size, image_size))
            test_images.append(new_array.reshape(-1, image_size, image_size, third_dimension))    
            test_image_names.append(image)
        except:
            pass
    
    # show model summary
    if(model_summary):
        model.summary()

    predictions = []
    for image, name in zip(test_images,test_image_names):
        prediction = model.predict(image)
        prediction_class = np.argmax(prediction)
        predictions.append("{0} : {1}".format(name,prediction_class))
        print("{0} : {1}".format(name,prediction_class))

        if(show_images):
            abs_path = os.path.join(images_path, name)
            img = cv2.imread(abs_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgplot = plt.imshow(img)
            plt.show()

    return predictions










