import numpy as np    
import cv2    
from sklearn.cluster import MeanShift, KMeans

for i in range(0, 6):
    #Loading original image
    inputFileName = './Images/'+str(i)+'.jpg'
    originImg = cv2.imread(inputFileName)

    # Shape of original image    
    originShape = originImg.shape

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities
    LabImg = cv2.cvtColor(originImg, cv2.COLOR_BGR2Lab)
    flatImg=np.reshape(LabImg, [-1, 3])

    ms = MeanShift(bandwidth = 12, bin_seeding=True, n_jobs=-1)

    # Performing meanshift on flatImg    
    ms.fit(flatImg)

    # (r,g,b) vectors corresponding to the different clusters after meanshift    
    labels=ms.labels_

    # Remaining colors after meanshift    
    cluster_centers = ms.cluster_centers_    

    # Finding and diplaying the number of clusters    
    labels_unique = np.unique(labels)    
    n_clusters_ = len(labels_unique)    
    print("number of estimated clusters : %d" % n_clusters_)

    # Displaying segmented image    
    '''
    segmentedImg = np.reshape(labels, originShape[:2])    
    cv2.imshow('Image',segmentedImg)    
    cv2.waitKey(0)    
    cv2.destroyAllWindows()
    '''

    segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
    segmentedImg = cv2.cvtColor(segmentedImg.astype(np.uint8), cv2.COLOR_Lab2BGR)
    outputFileName = './Images/'+str(i)+'-MS-'+str(n_clusters_)+'.jpg'
    cv2.imwrite(outputFileName, segmentedImg)




    flatImg=np.reshape(LabImg, [-1, 3])
    km = KMeans(n_clusters=n_clusters_)

    km.fit(flatImg)
    labels=km.labels_
    cluster_centers = km.cluster_centers_

    segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
    segmentedImg = cv2.cvtColor(segmentedImg.astype(np.uint8), cv2.COLOR_Lab2BGR)
    outputFileName = './Images/'+str(i)+'-KM-'+str(n_clusters_)+'.jpg'
    cv2.imwrite(outputFileName, segmentedImg)