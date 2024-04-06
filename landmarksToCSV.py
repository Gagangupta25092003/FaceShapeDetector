from facialLandmarkDetector import FacialLandmarkDetection
import os
import numpy as np

label = {
    "Heart" : 0,
    "Oblong" : 1,
    "Oval" : 2,
    "Round" : 3,
    "Square" : 4
}

def generate(folder_path):
    data = []
    labels = []
    for category in os.listdir(folder_path):              
        category_folder = os.path.join(folder_path, category)
        if  not (os.path.isdir(category_folder)):
            continue
        id = label[category]  
        
        for image in os.listdir(category_folder):
            try:
                imagePath = os.path.join(category_folder, image)
                Detector = FacialLandmarkDetection(imagePath=imagePath)
                points = Detector.load()
                data.append(points)
                labels.append(id)
            except Exception as e:
                print(e)
                continue
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

folder_path = "FaceShape Dataset"
training_folder_path = os.path.join(folder_path, "training_set")
testing_folder_path = os.path.join(folder_path, "testing_set")

training_data, training_labels = generate(training_folder_path)
testing_data, testing_labels = generate(testing_folder_path)
print("Training Data", training_data.shape)
print("Training Labels", training_labels.shape)
print("Testing Labels", testing_data.shape)
print("Testing Labels", testing_labels.shape)

np.savetxt("Training_data.csv", training_data.flatten(), delimiter=',')
np.savetxt("Training_labels.csv", training_labels.flatten(), delimiter=',')
np.savetxt("Testing_data.csv", testing_data.flatten(), delimiter=',')
np.savetxt("Testing_labels.csv", testing_labels.flatten(), delimiter=',')
