###
### This homework is modified from CS231.
###


import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # TODO: Implement this method!
    Z = np.zeros((3, 3))
    W = np.zeros((3, 3))
    Z[0][1], Z[1][0] = 1, -1
    W[0][1], W[1][0], W[2][2] = -1, 1, 1
    U, sigma, VT = np.linalg.svd(E)
    M = np.matmul(np.matmul(U, Z), U.T)
    Q1 = np.matmul(np.matmul(U, W), VT)
    Q2 = np.matmul(np.matmul(U, W.T), VT)
    R1 = np.linalg.det(Q1) * Q1
    R2 = np.linalg.det(Q2) * Q2
    T1 = U[:, 2]
    T2 = -U[:, 2]
    # print(f'R2 = {R2.shape}')
    # print(f'T1 = {T1.shape}')
    # print(np.expand_dims(T1, axis=1))
    # print(np.concatenate([R2, np.expand_dims(T1, axis=1)], axis=1))
    R1T1 = np.concatenate([R1, np.expand_dims(T1, axis=1)], axis=1)
    R1T2 = np.concatenate([R1, np.expand_dims(T2, axis=1)], axis=1)
    R2T1 = np.concatenate([R2, np.expand_dims(T1, axis=1)], axis=1)
    R2T2 = np.concatenate([R2, np.expand_dims(T2, axis=1)], axis=1)
    # print(np.array([R1T1, R1T2, R2T1, R2T2]).shape)
    return np.array([R1T1, R1T2, R2T1, R2T2])


'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    # image_points is p, size = (2, 2)
    # camera_matrices is M, size = (2, 3, 4)
    M = deepcopy(camera_matrices)
    n = M.shape[0]
    p = deepcopy(image_points)
    # print(f'M = {M}, shape = {M.shape}')
    # print(f'p = {p}, shape = {p.shape}')

    mat = np.zeros((n * 2, 4))
    for i in range(0, n):
        # print(i * 2, i * 2 + 1)
        mat[i * 2] = p[i, 1] * M[i, 2] - M[i, 1]
        mat[i * 2 + 1] = M[i, 0] - p[i, 0] * M[i, 2]
    # print(f'mat = {mat}')
    U, sigma, VT = np.linalg.svd(mat)
    # print(f'VT = {VT}')
    P_temp = VT[-1]
    P_temp /= P_temp[-1]
    # print(f'P_temp = {P_temp}')
    return P_temp[:3]
    
'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2M reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # TODO: Implement this method!
    
    # print(point_3d.shape)
    # print(image_points.shape)
    # print(camera_matrices.shape)
    M = deepcopy(camera_matrices)
    P = deepcopy(point_3d)
    P = np.append(P, 1)
    p = deepcopy(image_points)
    err = []
    for i in range(M.shape[0]):
        yi = np.dot(M[i], P)
        pi_prime = np.array([yi[0], yi[1]]) / yi[2]
        ei = pi_prime - p[i]
        err.extend(list(ei))
    # print(err)
    return np.array(err)

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # TODO: Implement this method!
    P = np.append(point_3d, 1)
    M = deepcopy(camera_matrices)
    # print(P.shape)
    # print(M.shape)
    Jac = np.zeros((2 * M.shape[0], 3))
    J_row = []
    for i in range(M.shape[0]):
        Mi = M[i]
        yi = np.matmul(Mi, P)
        # pi_prime = np.array([yi[0], yi[1]]) / yi[2]
        J_row.append((Mi[0, :3] * yi[2] - Mi[2, :3] * yi[0]) / yi[2] ** 2)
        J_row.append((Mi[1, :3] * yi[2] - Mi[2, :3] * yi[1]) / yi[2] ** 2)

    # print(J_row)
    for i in range(M.shape[0] * 2):
        Jac[i] = J_row[i]
    
    return Jac

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    P = linear_estimate_3d_point(image_points, camera_matrices)
    for _ in range(10):
        J = jacobian(point_3d=P, camera_matrices=camera_matrices)
        e = reprojection_error(point_3d=P, image_points=image_points, \
            camera_matrices=camera_matrices)
        P = P - np.matmul(np.matmul(np.linalg.inv(np.matmul(J.T, J)), J.T), e)
    return P
    
'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # TODO: Implement this method!
    init_RT = estimate_initial_RT(E)       # 4 * 3 * 4
    temp = np.matmul(K, np.hstack((np.eye(3), np.zeros((3,1)))))    # 3 * 4
    # print(M.shape)
    camera_matrices = np.array((temp, np.zeros(temp.shape)))
    # print(camera_matrices.shape)
    cnt_list = []
    for i in range(init_RT.shape[0]):
        camera_matrices[1] = np.matmul(K, init_RT[i])
        # print(camera_matrices)
        for j in range(image_points.shape[0]):
            cnt = 0
            # print(i, j)
            nonlinear_pt = nonlinear_estimate_3d_point(image_points[j], \
                camera_matrices)
            nonlinear_pt = np.append(nonlinear_pt, 1)
            # print(nonlinear_pt)
            # print(np.vstack((init_RT[i], [0, 0, 0, 1])))
            temp2 = np.vstack((init_RT[i], [0, 0, 0, 1]))
            # print(np.array((nonlinear_pt[0], nonlinear_pt[1], nonlinear_pt[2], 1)))
            Pj_prime = np.matmul(temp2, np.array((nonlinear_pt[0], nonlinear_pt[1], nonlinear_pt[2], 1)).T)
            Pj_prime /= Pj_prime[3]
            Pj_prime = Pj_prime[:3]
            # print(f'Pj = {nonlinear_pt}')
            # print(f'Pj_prime = {Pj_prime}')
            if nonlinear_pt[2] > 0 and Pj_prime[2] > 0:
                cnt += 1
        cnt_list.append(cnt)
    # print(cnt_list.index(max(cnt_list)))
    return init_RT[cnt_list.index(max(cnt_list))]
    

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    # original_camera_matrices = camera_matrices
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())

    # tolerance = 0  # Adjust the tolerance as needed
    # are_same = np.allclose(original_camera_matrices, camera_matrices, rtol=tolerance, atol=tolerance)
    
    # print("after, the camera_metrics is the same ", are_same)
    # print("expected_3d_point: ", estimated_3d_point)

    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    np.save('results.npy', dense_structure)
    print ('Save results to results.npy!')