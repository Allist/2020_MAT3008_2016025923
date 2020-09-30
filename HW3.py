import numpy as np
import random

points = np.array([[-2.9, 35.3], 
                   [-2.1, 19.7], 
                   [-0.9, 5.7], 
                   [1.1, 2.1], 
                   [0.1, 1.2], 
                   [1.9, 8.7], 
                   [3.1, 25.7], 
                   [4.0, 41.5]])

def pop_six_random_points():
    global points

    ret_six_points = points
    ret_six_points = np.delete(ret_six_points, random.randint(0, 7), 0)
    ret_six_points = np.delete(ret_six_points, random.randint(0, 6), 0)
    print('---- select six sample points ----')
    print(ret_six_points)

    return ret_six_points

def predict_curve(sample_points):
    mat_A = np.array([])
    for i in range(len(sample_points)):
        tmp = np.array([sample_points[i, 0]**2, 
                        sample_points[i, 0], 
                        1])
        mat_A = np.append(mat_A, tmp)
    mat_A = mat_A.reshape(int(len(mat_A)/3), 3)
    print('---- matrix A ----')
    print(mat_A)

    curve = np.linalg.inv(mat_A.T@mat_A)@mat_A.T@sample_points[:, 1]
    print('---- result of curve fitting [a, b, c] ----')
    print(curve)

def main():
    sample_points = pop_six_random_points()
    predict_curve(sample_points)

if __name__ == "__main__":
    main()