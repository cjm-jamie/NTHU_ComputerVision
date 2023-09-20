import numpy as np

def my_imfilter(image, imfilter):
    """function which imitates the default behavior of the build in scipy.misc.imfilter function.

    Input:
        image: A 3d array represent the input image.
        imfilter: The gaussian filter.
    Output:
        output: The filtered image.
    """
    # =================================================================================
    # TODO:                                                                           
    # This function is intended to behave like the scipy.ndimage.filters.correlate    
    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         
    # of the filter matrix.)                                                          
    # Your function should work for color images. Simply filter each color            
    # channel independently.                                                          
    # Your function should work for filters of any width and height                   
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This       
    # restriction makes it unambigious which pixel in the filter is the center        
    # pixel.                                                                          
    # Boundary handling can be tricky. The filter can't be centered on pixels         
    # at the image boundary without parts of the filter being out of bounds. You      
    # should simply recreate the default behavior of scipy.signal.convolve2d --       
    # pad the input image with zeros, and return a filtered image which matches the   
    # input resolution. A better approach is to mirror the image content over the     
    # boundaries for padding.                                                         
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can 
    # see the desired behavior.                                                       
    # When you write your actual solution, you can't use the convolution functions    
    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   
    # Simply loop over all the pixels and do the actual computation.                  
    # It might be slow.                        
    
    # NOTE:                                                                           
    # Some useful functions:                                                        
    #     numpy.pad (https://numpy.org/doc/stable/reference/generated/numpy.pad.html)      
    #     numpy.sum (https://numpy.org/doc/stable/reference/generated/numpy.sum.html)                                     
    # =================================================================================

    # ============================== Start OF YOUR CODE ===============================

    output = np.zeros_like(image)
    
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]

    H_image = image.shape[0]
    W_image = image.shape[1]
    H_imfilter = imfilter.shape[0]
    W_imfilter = imfilter.shape[1]

    # padding size according to kernal, processing boundary issue of the image
    v_offset = (H_imfilter-1) // 2
    h_offset = (W_imfilter-1) // 2

    # padding three channel with vertical and horizontal offset
    R_pad = np.pad(R, (v_offset, h_offset), mode='constant')
    G_pad = np.pad(G, (v_offset, h_offset), mode='constant')
    B_pad = np.pad(B, (v_offset, h_offset), mode='constant')

    # do convolution for loop for each pixel in the image
    for h in range(H_image):
        for w in range(W_image):
            output[h][w][0] = np.sum(np.multiply(R_pad[h:h+H_imfilter, w:w+W_imfilter], imfilter))
            output[h][w][1] = np.sum(np.multiply(G_pad[h:h+H_imfilter, w:w+W_imfilter], imfilter))
            output[h][w][2] = np.sum(np.multiply(B_pad[h:h+H_imfilter, w:w+W_imfilter], imfilter))

    print("I'm here!")

    # =============================== END OF YOUR CODE ================================

    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can
    # see the desired behavior.
    # import scipy.ndimage as ndimage
    # output = np.zeros_like(image)
    # for ch in range(image.shape[2]):
    #    output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')

    return output