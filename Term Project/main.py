import sys, copy
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.markers as markers
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.stats
from sklearn.decomposition import PCA
import cv2

g_dim = 3
g_train_cnt_per_cluster = 300
g_pred_cnt_per_cluster = 100
g_cluster_cnt = 5

def create_points(means, stddevs, sample_cnt):
    data = np.zeros(shape=(sample_cnt,g_dim))
    for idx in range(0, len(means)):
        rands = np.random.normal(means[idx], stddevs[idx], sample_cnt)
        data[:, idx] = rands

    return data

def draw_points(points_cluster_group, mean, box_points = None):
    # black color means out of cluster
    colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#000000', '#00ffff']
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax = fig.add_subplot(111, projection='3d')
    for i in range(0, g_cluster_cnt+1):
        ax.scatter(points_cluster_group[i][:, 0], 
                    points_cluster_group[i][:, 1], 
                    points_cluster_group[i][:, 2], 
                    c=colors[i])

    # draw mean points
    ax.plot(mean[:, 0], mean[:, 1], mean[:, 2], '*', c=colors[-1], markeredgecolor=colors[-2], markersize=20)

    # draw box shapes
    if(type(box_points) != type(None)):
        for i in range(0, g_cluster_cnt):
            '''
            Z = np.array(np.meshgrid(intervals[i][0], intervals[i][1], intervals[i][2])).T.reshape(-1,3)
            verts = [[Z[0],Z[1],Z[3],Z[2]], #XY
            [Z[4],Z[5],Z[7],Z[6]], #XY
            [Z[0],Z[1],Z[5],Z[4]], #YZ
            [Z[2],Z[3],Z[7],Z[6]], #YZ
            [Z[0],Z[2],Z[6],Z[4]], #XZ
            [Z[1],Z[3],Z[7],Z[5]]] #XZ
            '''
            ax.add_collection3d(Poly3DCollection(box_points[i], facecolors=colors[i], linewidths=1, edgecolors=colors[i], alpha=.1))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

# intervals: g_cluster_cnt * g_dim * 2
# return: g_cluster_cnt * 6 * 4 * 3
def make_interval_box_points(intervals):
    ret = np.empty(shape=(g_cluster_cnt, 6, 4, 3))
    for i in range(0, g_cluster_cnt):
        Z = np.array(np.meshgrid(intervals[i][0], intervals[i][1], intervals[i][2])).T.reshape(-1,3)
        ret[i] = [[Z[0],Z[1],Z[3],Z[2]], #XY
                [Z[4],Z[5],Z[7],Z[6]], #XY
                [Z[0],Z[1],Z[5],Z[4]], #YZ
                [Z[2],Z[3],Z[7],Z[6]], #YZ
                [Z[0],Z[2],Z[6],Z[4]], #XZ
                [Z[1],Z[3],Z[7],Z[5]]] #XZ
    return ret

# shape of mean = [1, g_dim]
# shape of samples = [group of cluster cnt, g_dim]
def calc_section(mean, samples, confidence = 0.98):
    stddevs = np.std(samples, axis=0)
    intervals = np.array(scipy.stats.norm.interval(confidence, loc=mean, scale=stddevs))
    return intervals.T

def doTermProject():
    means = np.array(  [[3,1,8], 
                        [10,10,-1], 
                        [5,-5,0], 
                        [-10,-10,1], 
#                        [-5,5,2], 
                        [0,0,-12], 
                        ])
    stddevs = np.array([[2,2,5], 
                        [2,3,1], 
                        [3,3,2], 
                        [1,5,3], 
                        [3,3,3], 
                        ])

    # Make training samples
    train_samples = np.zeros(shape=(g_cluster_cnt*g_train_cnt_per_cluster, g_dim))
    for col_idx in range(0, means.shape[0]):
        train_samples[col_idx*g_train_cnt_per_cluster:(col_idx+1)*g_train_cnt_per_cluster, :] = \
         create_points(means[col_idx], stddevs[col_idx], g_train_cnt_per_cluster)

    # Make cluster group of sample points
    train_sample_groups = [None]*(g_cluster_cnt+1)
    for i in range(0, g_cluster_cnt):
        train_sample_groups[i] = train_samples[g_train_cnt_per_cluster*i:g_train_cnt_per_cluster*(i+1), :]
    train_sample_groups[5] = np.zeros(shape=(0,g_dim))
    # Draw plot that cluster group of sample points
    # draw_points(train_sample_groups)

    # Do K-means clustering
    km = KMeans(n_clusters=g_cluster_cnt)
    km.fit(train_samples)
    labels = km.labels_
    centers = km.cluster_centers_

    # Make cluster group of sample points using labels
    train_sample_groups = [np.zeros(shape=(0, g_dim))]*(g_cluster_cnt+1)
    for i in range(0, len(labels)):
        train_sample_groups[labels[i]] = np.vstack((train_sample_groups[labels[i]], train_samples[i, :]))
    # draw_points(train_sample_groups)

    # Get cluster boundary
    intervals = np.zeros(shape=(g_cluster_cnt, g_dim, 2))
    for i in range(0, g_cluster_cnt):
        intervals[i] = calc_section(centers[i], train_sample_groups[i])
    box_points = make_interval_box_points(intervals)

    # Make Predict samples
    means = np.vstack((means, np.array([-5, 5, 2])))
    stddevs = np.vstack((stddevs, np.array([4,1,2])))
    pred_samples = np.zeros(shape=((g_cluster_cnt+1)*g_pred_cnt_per_cluster, g_dim)) # add 1 cluster
    for col_idx in range(0, means.shape[0]):
        pred_samples[col_idx*g_pred_cnt_per_cluster:(col_idx+1)*g_pred_cnt_per_cluster, :] = \
            create_points(means[col_idx], stddevs[col_idx], g_pred_cnt_per_cluster)
    pca_pred_samples = copy.deepcopy(pred_samples)
    
    # Do predict
    labels = km.predict(pred_samples)
    pca_labels = copy.deepcopy(labels)

    # Print Matching table
    matching_tbl = np.full(6, -1,dtype=int)
    matching_tbl[5] = 5
    # matching_tbl = np.empty(shape=(5)).fill(-1)
    for i in range(0, g_cluster_cnt):
        optimal_dist = sys.float_info.max
        label = -1
        for j in range(0, g_cluster_cnt):
            if(i == j):
                continue
            dist = np.linalg.norm(means[i]-centers[j])
            if(dist < optimal_dist):
                optimal_dist = dist
                label = j
        matching_tbl[label] = i
    
    # Check same exist, then restart
    for i in range(0, g_cluster_cnt):
        if(matching_tbl[i] == -1):
            #restart
            return False
    
    # Check predict rate
    matching_cnt = 0
    for i in range(0, len(labels)):
        if(matching_tbl[labels[i]] == int(i/100)):
            matching_cnt = matching_cnt + 1
    print('Only using K-means clustering, predict rate: ', matching_cnt/(g_cluster_cnt+1)/g_pred_cnt_per_cluster)

    # Make cluster group of predict sample points using labels
    pred_sample_groups = [np.zeros(shape=(0, g_dim))]*(g_cluster_cnt+1)
    for i in range(0, len(labels)):
        pred_sample_groups[labels[i]] = np.vstack((pred_sample_groups[labels[i]], pred_samples[i, :]))
    pca_pred_sample_groups = copy.deepcopy(pred_sample_groups)
    draw_points(pred_sample_groups, centers, box_points)

    # Check each point belongs to cluster or not.
    # Find out of clusters
    for i in range(0, len(labels)):
        cluster_idx = labels[i]
        sample = pred_samples[i]
        for d in range(0, g_dim):
            if(sample[d] < intervals[cluster_idx][d][0] or intervals[cluster_idx][d][1] < sample[d]):
                #cluster_idx = g_cluster_cnt
                labels[i] = g_cluster_cnt
                break
    
    # Check predict rate
    matching_cnt = 0
    for i in range(0, len(labels)):
        if(matching_tbl[labels[i]] == int(i/100)):
            matching_cnt = matching_cnt + 1
    print('Using K-means and interval, predict rate: ', matching_cnt/(g_cluster_cnt+1)/g_pred_cnt_per_cluster)

    # Make cluster group of predict sample points using labels
    pred_sample_groups = [np.zeros(shape=(0, g_dim))]*(g_cluster_cnt+1)
    for i in range(0, len(labels)):
        pred_sample_groups[labels[i]] = np.vstack((pred_sample_groups[labels[i]], pred_samples[i, :]))
    print('When using K-means and interval, Out of cluster cnt: ', len(pred_sample_groups[g_cluster_cnt]))
    draw_points(pred_sample_groups, centers, box_points)

    # Do PCA
    pca_list = [None] * g_cluster_cnt
    pca_intervals = np.zeros(shape=(g_cluster_cnt, g_dim, 2))
    for i in range(0, g_cluster_cnt):
        pca_list[i] = PCA(n_components=3)
        pca_list[i].fit(train_sample_groups[i])
        pca_center = pca_list[i].transform(centers[i].reshape(1, -1)).reshape(3)
        #pca_train_sample_groups[i] = pca_list[i].transform(train_sample_groups[i])
        #pca_intervals[i] = calc_section(pca_centers[i], pca_train_sample_groups[i])
        pca_intervals[i] = calc_section(pca_center, pca_list[i].transform(train_sample_groups[i]))
    
    for i in range(0, len(pca_labels)):
        cluster_idx = pca_labels[i]
        sample = pca_list[pca_labels[i]].transform(pca_pred_samples[i].reshape(1, -1)).reshape(3)
        for d in range(0, g_dim):
            if(sample[d] < pca_intervals[cluster_idx][d][0] or pca_intervals[cluster_idx][d][1] < sample[d]):
                #cluster_idx = g_cluster_cnt
                pca_labels[i] = g_cluster_cnt
                break
    
    box_points = make_interval_box_points(pca_intervals)
    for i in range(0, g_cluster_cnt):
        box_points[i] = pca_list[i].inverse_transform(box_points[i].reshape(24,3)).reshape(6,4,3)
    draw_points(train_sample_groups, centers, box_points)

    matching_cnt = 0
    for i in range(0, len(pca_labels)):
        if(matching_tbl[pca_labels[i]] == int(i/100)):
            matching_cnt = matching_cnt + 1
    print('Using K-means clustering and intervals in pca, predict rate: ', matching_cnt/(g_cluster_cnt+1)/g_pred_cnt_per_cluster)

    # Make cluster group of predict sample points using labels
    pca_pred_sample_groups = [np.zeros(shape=(0, g_dim))]*(g_cluster_cnt+1)
    for i in range(0, len(pca_labels)):
        pca_pred_sample_groups[pca_labels[i]] = np.vstack((pca_pred_sample_groups[pca_labels[i]], pca_pred_samples[i, :]))
    print('Using K-means clustering and intervals in pca, Out of cluster cnt: ', len(pca_pred_sample_groups[g_cluster_cnt]))
    draw_points(pca_pred_sample_groups, centers, box_points)


    '''
    # Do PCA
    pca_list = [None] * g_cluster_cnt
    pca_intervals = np.zeros(shape=(g_cluster_cnt, g_dim, 2))
    pca_train_sample_groups = train_sample_groups
    pca_centers = [None] * g_cluster_cnt
    real_pca_pred_sample_groups = [np.zeros(shape=(0, g_dim))]*(g_cluster_cnt+1)
    for i in range(0, g_cluster_cnt):
        pca_list[i] = PCA(n_components=3)
        pca_list[i].fit(train_sample_groups[i])
        pca_centers[i] = pca_list[i].transform(centers[i].reshape(1, -1)).reshape(3)
        #pca_train_sample_groups[i] = pca_list[i].transform(train_sample_groups[i])
        #pca_intervals[i] = calc_section(pca_centers[i], pca_train_sample_groups[i])
        pca_intervals[i] = calc_section(pca_centers[i], pca_list[i].transform(train_sample_groups[i]))

        pca_pred_sample_groups[i] = pca_list[i].transform(pca_pred_sample_groups[i])
        for j in range(0, pca_pred_sample_groups[i].shape[0]):
            sample = pca_pred_sample_groups[i][j]
            isOut = False
            for d in range(0, g_dim):
                if(sample[d] < pca_intervals[i][d][0] or pca_intervals[i][d][1] < sample[d]):
                    isOut = True
                    break
            sample = pca_list[i].inverse_transform(sample.reshape(1, -1)).reshape(-1)
            if(isOut):
                real_pca_pred_sample_groups[g_cluster_cnt] = np.vstack((real_pca_pred_sample_groups[g_cluster_cnt], sample))
            else:
                real_pca_pred_sample_groups[i] = np.vstack((real_pca_pred_sample_groups[i], sample))

    box_points = make_interval_box_points(pca_intervals)
    for i in range(0, g_cluster_cnt):
        box_points[i] = pca_list[i].inverse_transform(box_points[i].reshape(24,3)).reshape(6,4,3)
    draw_points(train_sample_groups, centers, box_points)
    draw_points(real_pca_pred_sample_groups, centers, box_points)
    '''
    

    return True

def main():
    flag = False
    while (flag == False):
        # To find proper samples...
        flag = doTermProject()
    

if __name__ == "__main__":
    main()