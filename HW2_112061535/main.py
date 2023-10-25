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
    # Step 1: SVD to factorize the essential matrix
    U, D, Vt = np.linalg.svd(E)
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    Q1 = U @ W @ Vt
    Q2 = U @ W.T @ Vt
    R1 = np.linalg.det(Q1) * Q1
    R2 = np.linalg.det(Q2) * Q2

    # Step 2: Calculate the translation
    u3 = U[:, 2]
    T1 = u3
    T2 = -u3

    # Step 3: Combine R and T to get the four combinations of R and T
    RT1 = np.hstack((R1, T1[:, np.newaxis]))
    RT2 = np.hstack((R1, T2[:, np.newaxis]))
    RT3 = np.hstack((R2, T1[:, np.newaxis]))
    RT4 = np.hstack((R2, T2[:, np.newaxis]))

    estimated_RTs = np.array([RT1, RT2, RT3, RT4])
    
    return estimated_RTs

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
    num_cameras = camera_matrices.shape[0]

    # Create the DLT matrix A
    # each camera will have 2 eqns
    # 4 for homogeneous coordinates of a 3D point (X, Y, Z, W)
    A = np.zeros((2 * num_cameras, 4))
    for i in range(num_cameras):
        M = camera_matrices[i]
        u, v = image_points[i]
        # v * Mi^3 − Mi^2
        A[2 * i, :] = v * M[2, :] - M[1, :]
        # Mi^1 - u * Mi^3
        A[2 * i + 1, :] = M[0, :] - u * M[2, :]

    # Solve for the 3D point using SVD
    _, _, Vt = np.linalg.svd(A)
    estimated_3d_point_homogeneous = Vt[-1, :] / Vt[-1, -1]

    # Convert to Cartesian coordinates (remove the last element, which is W)
    estimated_3d_point = estimated_3d_point_homogeneous[:-1]

    return estimated_3d_point
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
    num_cameras = camera_matrices.shape[0]
    error = np.zeros(2 * num_cameras)  # Initialize the error vector.
    P = np.append(point_3d, 1)

    for i in range(num_cameras):
        Mi = camera_matrices[i]
        yi = np.dot(Mi, P) # Project 3D point P into image coordinates.
        p_prime_x, p_prime_y = np.array([yi[0], yi[1]]) / yi[2]

        ui, vi = image_points[i]

        # Compute the reprojection error for both u and v.
        error[2*i] = p_prime_x - ui
        error[2*i + 1] = p_prime_y - vi

    return np.array(error)

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
    # print("P.shape is ", P.shape)
    num_cameras = camera_matrices.shape[0]
    # jac = np.zeros((2 * num_cameras, 3))  # Initialize the Jacobian matrix.
    jac = []

    for i in range(num_cameras):
        Mi = camera_matrices[i]
        yi = np.dot(Mi, P)

        jac.append((Mi[0, :3] * yi[2] - Mi[2, :3] * yi[0]) / yi[2] ** 2)
        jac.append((Mi[1, :3] * yi[2] - Mi[2, :3] * yi[1]) / yi[2] ** 2)

    return np.array(jac)

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
    iteration_time = 10

    for _ in range(iteration_time):
        J = jacobian(P, camera_matrices)
        e = reprojection_error(P, image_points, camera_matrices)
        P = P - np.dot(np.dot(np.linalg.inv(np.dot(J.T, J)), J.T), e)
    return P
'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
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
    init_RT = estimate_initial_RT(E)       # (4, 3, 4)
    # updates the camera matrix for the first camera 
    matrices_1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3,1)))))    # 3 * 4

    max_positive_z_count = 0
    best_RT = None

    camera_matrices = np.array((matrices_1, np.zeros(matrices_1.shape)))
    for i in range(init_RT.shape[0]):
        # updates the camera matrix for the second camera 
        camera_matrices[1] = np.dot(K, init_RT[i])
        # image_points.shape[0] represent pt_num for each camera system
        for j in range(image_points.shape[0]):
            cnt = 0
            nonlinear_pt = nonlinear_estimate_3d_point(image_points[j], camera_matrices)
            # homogeneous coordinate
            nonlinear_pt = np.append(nonlinear_pt, 1)

            # camera matrix for second camera
            matrices_2 = np.vstack((init_RT[i], [0, 0, 0, 1]))

            projected_point = np.dot(matrices_2, np.array((nonlinear_pt[0], nonlinear_pt[1], nonlinear_pt[2], 1)).T)
            projected_point /= projected_point[3]
            projected_point = projected_point[:-1]
            # cnt = points which have positive z-coordinate for both images
            if nonlinear_pt[2] > 0 and projected_point[2] > 0:
                cnt += 1
        if cnt > max_positive_z_count:
            max_positive_z_count = cnt
            best_RT = init_RT[i]

    return best_RT

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
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    print("expected_3d_point: ", estimated_3d_point)
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
