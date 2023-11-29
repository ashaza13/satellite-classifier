import pandas
import numpy
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import cv2
import random

'training data should have 18000 no plane images, leaving six for testing'
'training data should have 6000 plane images, leaving 2000 for testing'

labels = []
all2test = []
count = 0
for filename in os.listdir("images"):
    path = os.path.join("images", filename)
    img = cv2.imread(path)
    if count < 18000:
        labels.append(0)
        count += 1
    elif count < 24000:
        labels.append(0)
        count += 1
    elif count < 30000:
        labels.append(1)
        count += 1
    else:
        labels.append(1)
    all2test.append(img)
    
images = numpy.array(all2test)
newLabels = numpy.array(labels)

images_flattened = images.reshape(images.shape[0], -1)

images_train, images_test, labels_train, labels_test = train_test_split(images_flattened, newLabels, test_size=0.2, shuffle=True, stratify=newLabels)

classifier = SVC()

parameters = [{'gamma': [0.01, 0.1, 1], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(images_train, labels_train)

best_estimator = grid_search.best_estimator_

prediction = best_estimator.predict(images_test)

score = accuracy_score(prediction, labels_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))




