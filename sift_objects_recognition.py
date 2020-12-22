import fnmatch
import shutil
from collections import Counter
from shutil import Error
import os
import cv2
import pickle

import numpy as np

filename = "keypoints.p"
model = "model"
descriptors = " descriptors.csv"
model_test = " keypoint_test"
number_descriptor = 200


def detector_shift_training_base(dataset_path="./training"):
    images_train = {}
    descs = []
    descriptor = []
    classes = []
    errors = []
    for path, dirs, files in os.walk(dataset_path):
        for file in files:
            if fnmatch.fnmatch(file, '*.jpg'):
                fullname = os.path.join(path, file)
                classe = os.path.basename(path)
                img = cv2.imread(fullname, 0)
                # height, weight, channels = img.shape
                # print(height, weight, channels)
                """image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)"""
                images_train['image'] = img
                images_train['classe'] = classe
                classes.append(classe)
                sift = cv2.SIFT_create(number_descriptor)
                #sift = cv2.SIFT_create()
                # Take the first element of the dictionary to compute the keypoints and descriptor of the image
                keypoints, train_descriptor = sift.detectAndCompute(images_train['image'], None)
                descriptor.append(train_descriptor)
                sift_keypoints = [train_descriptor, classe]
                descs.append(sift_keypoints)

                i = 0
                keypoint = [train_descriptor, classe]
                for point in keypoints:
                    temp = (point.pt, point.size, point.angle, point.response, point.octave,
                            point.class_id)
                    ++i
                    keypoint.append(temp)
                # Keypoints with descriptor and classe of images and keypoints stored
                with open(model, "wb+") as fichier:
                    pickle.dump(keypoint, fichier)
                # Store keypoints and descriptor into a file called model
                if not os.path.exists("keypoints_train"):
                    os.mkdir("keypoints_train")
                    print("folder Created ", "keypoints")
                else:
                    break
                try:
                    shutil.move(filename, "model_keypoints")
                except Error as err:
                    errors.extend(err.args[0])
    # Keypoints with descriptor and classe of images
    with open('base_keypoints', 'wb+') as file:
        pickle.dump([descriptor, classes], file)

    with open(filename, "wb+") as base:
        pickle.dump(descs, base)


def decriptor_test_image(filename):
    des = pickle.load(open(filename, 'rb'))
    print(des)
    errors = []
    # Create sift object from Sift class
    sift = cv2.SIFT_create(number_descriptor)
    #sift = cv2.SIFT_create()
    # Joins the home directory to the users typed in folder
    user_input = input(" Type the path of the image followed by the image : ")
    classe_name = os.path.dirname(os.path.abspath(user_input))
    classe = os.path.basename(classe_name)
    print(classe)
    img = cv2.imread(user_input)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    keypoints, descriptor = sift.detectAndCompute(image_gray, None)
    cv2.imshow('original', img)
    img_with_keypoints = cv2.drawKeypoints(image_gray, keypoints, image_gray,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('keypoints', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    descriptor_test = [descriptor, classe]

    # Keypoints with descriptor and classe of images
    with open("Descriptor_test", "wb+") as base:
        pickle.dump(descriptor_test, base)

    # print(keypoints, descriptor)
    i = 0
    keypoint = [descriptor, classe]
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        ++i
        keypoint.append(temp)

        with open(model_test, "wb+") as fichier:
            pickle.dump(keypoint, fichier)
        # Store keypoints and descriptor into a file called model
        if not os.path.exists("Keypoint_test"):
            os.mkdir("Keypoint_test")
            print("folder Created ", "keypoints")
        else:
            break
        try:
            shutil.move(model_test, "Keypoint_test")
        except Error as err:
            errors.extend(err.args[0])

    return descriptor_test


def matching(descriptor, descs, classes, k=3):
    correct = 0
    total = 0
    k_flann = 2
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=200)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    M = []
    if len(descriptor) > k_flann:
        tmp = []
        for idy, v in enumerate(descs):
            c1 = 0
            matches = flann.knnMatch(v, descriptor, k=k_flann)
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    c1 += 1
            tmp.append(c1)
            M.append([idy, c1])
        M.sort(key=lambda x: x[1], reverse=True)

        k_nearest = M[:k]
        E = []
        for b in k_nearest:
            E.append(b[0])
        dav = Counter(E)

        predict = dav.most_common(1)[0][0]
        if classes[predict] == classes:
            correct +=1
        total +=1

        print(" La classe predicte de l'image est  : ", classes[predict])
        # print("le taux de precision overall", round((100 * correct / total), 2), ' %')
        return predict, classes[predict]


def write_classes_unique():
    classes = []
    for doss in os.listdir("training"):
        classes.append(doss)
    pickle.dump(classes, open("classes.cl", "wb"))


def predict(filename, model='base_keypoints', show=False):
    # loading of descriptors's training
    des = pickle.load(open(model, 'rb'))
    descriptors = des[0]
    classes = des[1]
    classe_name = os.path.dirname(os.path.abspath(filename))
    classe = os.path.basename(classe_name)
    # print(classe)
    # Extracting the descriptors of input image
    # Create sift object from Sift class
    #sift = cv2.SIFT_create()
    sift = cv2.SIFT_create(number_descriptor)
    img = cv2.imread(filename, 0)

    keypoints, descriptor = sift.detectAndCompute(img, None)
    m = matching(descriptor, descriptors, classes)
    if show:
        cv2.imshow('Image test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Matching function to predict the image test
    return m


def test(dataset_path="./test"):
    correct = 0
    total = 0
    classes = pickle.load(open("classes.cl", "rb"))
    print(isinstance(classes, list))
    matrice = np.zeros((len(classes), len(classes)))
    for path, dirs, files in os.walk(dataset_path):
        for file in files:
            if fnmatch.fnmatch(file, '*.jpg'):
                fullname = os.path.join(path, file)
                classe = os.path.basename(path)
                id_correct = classes.index(classe)
                pred = classes.index(predict(fullname)[1])
                if id_correct == pred:
                    correct += 1
                total += 1

                matrice[id_correct][pred] += 1
    print("overall precision ", round((100 * correct / total), 2), ' %')
    with open('matrice_confusion', 'wb+') as file:
        pickle.dump(matrice, file)
    # print(matrice)


detector_shift_training_base("training")
#predict('test/watch/12.jpg', show=True)
write_classes_unique()
#test(dataset_path="./test")
#decriptor_test_image('model')