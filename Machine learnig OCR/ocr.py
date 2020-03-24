import cv2
from glob import glob
from skimage import feature
import numpy as np
import time

path = "../notMNIST_small/notMNIST_small/"
HuMoments = ["HM1",
             "HM2",
             "HM3",
             "HM4",
             "HM5",
             "HM6",
             "HM7",
             "Hog1",
             "Hog2",
             "Hog3",
             "Hog4",
             "Hog5",
             "Hog6",
             "Hog7",
             "Hog8",
             "Hog9",
             "perim",
             "radius",
             "area",
             "Class"]


def extract_HuMoments(image):
    # Threshold image
    _, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    # Calculate Moments
    moments = cv2.moments(cnt)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    # perimetre
    epsilon = cv2.arcLength(cnt, True)
    # rayon
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    # area
    area = cv2.contourArea(cnt)

    return (epsilon*0.0001, radius*0.01, area*0.001, huMoments)


def get_hog_features(img, orient, pix_per_cell, cell_per_block):
    features = feature.hog(img, orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           block_norm="L1-sqrt",
                           visualize=False, feature_vector=True)
    return features


def writeHuMoments_Tofile(folderPath, File, Class):
    for fn in glob(folderPath):
        # Read image as grayscale image
        im = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

        start_time = time.time()
        # extract HuMoments features
        perim, radius, area, HuM_features = extract_HuMoments(im)
        # extract hog features
        Hog_features = get_hog_features(im, 9, 28, 1)

        HuMoments_F = (str(w[0]) for w in HuM_features)
        Hog_F = (str(w) for w in Hog_features)

        File.write(', '.join(HuMoments_F)+', ' +
                   ', '.join(Hog_F)+', %f, %f, %f, %d\n' % (perim, radius, area, Class))


print("Start")

# creating a file
file = open("csv/HuM_Hog_Features.csv", "w+")

file.write(', '.join(HuMoments)+'\n')

start = 65
end = start+10


start_time = time.time()
for i in range(start, end):
    print(chr(i)+"\n")
    folderPath = path+chr(i)+"/*.png"
    writeHuMoments_Tofile(folderPath, file, i-start+1)
end_time = time.time()-start_time
print("Temps d execution : %s secondes ---" % (end_time))

file.close()

print("End")
