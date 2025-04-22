### Multiclass Classification -
### Use CIFAR10 dataset and develop a ML model for image classification using kNN.
### CIFAR10 images (in PNG format) can be downloaded from: https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders

import os   # It helps in interacting with the operating system, like reading folders and file paths.
import numpy as np
from PIL import Image   # PIL (Python Imaging Library) lets us open, resize and convert image formats.
from sklearn.neighbors import KNeighborsClassifier    # Performs classification by comparing distances b/w samples.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_images(folder_path):
    data = []   # Stores the actual image arrays.
    labels = []   # Stores the class (label) for each image.
    class_folders = sorted(os.listdir(folder_path))    # os.listdir(): lists all files & folders in the given directory ; sorted(): ensures consistent alphabetical order of class names.
    for lbl_name in class_folders:   # Loop via each class (e.g. airplane, automobile, etc.)
        cls_path = os.path.join(folder_path, lbl_name)   # Combines the main folder with subfolder to get full path.
        if not os.path.isdir(cls_path):   # Checks if the current path is a directory, and not a file. If it's not a folder, then skip it.
            continue
        for file_name in os.listdir(cls_path):   # Loops through every image in class folder.
            img_path = os.path.join(cls_path, file_name)   # Constructs the full path of each image file.
            try:
                # Image.open() - opens the image file, .resize((32,32)) - makes sure all images are of the same size, .convert('RGB') - makes sure all images have 3 colour channels.
                img = Image.open(img_path).resize((32, 32)).convert("RGB")
                # .flatten() - turns the (32 - height, 32 - width, 3 - RGB) shape into a 1D array of 3072 values bec. KNN accepts 1D numerical vector.
                data.append(np.array(img).flatten())
                labels.append(lbl_name)   # Adds the class name to the labels list.
            except Exception as err:   # If any error occurs (e.g., file is corrupt), it skips that file.
                continue
    return np.array(data), np.array(labels)

def train_model(X_train, y_train, n_neighbors=6):   # n_neighbors: By default is 3, but it was giving a bit low accuracy in case of 3 and 5, so chose 6 as optimum value here.
    """
    :param X_train: image data
    :param y_train: corresponding labels (default=3)
    :param n_neighbors: how many neighbors to use
    :return:
    """
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)    # Creates a KNN classifier object using 3 nearest neighbors.
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    disp.plot(cmap='Blues')
    plt.show()
    return acc, report, cm

def main():
    trainset_path = "cifar10/train"
    testset_path = "cifar10/test"
    print("Wait! Loading images....")
    X_train, y_train = load_images(trainset_path)
    X_test, y_test = load_images(testset_path)
    print("Wait! Model training happening....")
    model = train_model(X_train, y_train, n_neighbors=6)
    print("Evaluating the model performance....")
    acc, report, cm = evaluate_model(model, X_test, y_test)
    print(f"Accuracy of KNN Model on CIFAR10 Dataset is: {acc*100:.2f}%")
    print(f"Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
