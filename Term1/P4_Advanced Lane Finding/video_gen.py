from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import glob
from calibrate_cam import calibrate_camera
from tracker import tracker

mtx, dist = calibrate_camera()

# Applies a Gaussian Noise kernel
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Calculate color transformations, image gradients and thresholds
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    #grad_binary = scaled_sobel
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255 * abs_sobelxy/np.max(abs_sobelxy))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    #grad_binary = scaled_sobel
    return grad_binary

# Define a color thresholding function using saturtion from HLS
# and value/brightness from HSV and return a binary image
def color_threshold(img, hthresh=(0, 180), sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls[:,:,0]
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= hthresh[0]) & (h_channel <= hthresh[1])] = 1
    #cv2.imshow('h_chnl', resize_frame(h_binary * 255))
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1
    #cv2.imshow('s_chnl', resize_frame(s_channel * 255))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1
    #cv2.imshow('v_chnl', resize_frame(v_channel * 255))
    output = np.zeros_like(s_channel)
    #output[(h_binary == 1) | (s_binary == 1) & (v_binary == 1)] = 1
    # Sagar's
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output
    
def get_src_dst_points(img):
    
    imgH, imgW = img.shape[:2]
    img_size = img.shape[:2]
    src = np.zeros((4, 2), dtype='float32') 
    dst = np.zeros((4, 2), dtype='float32')
    """
    # Sagar's
    src = np.float32([[585. /1280.*img_size[1], 455./720.*img_size[0]], [705. /1280.*img_size[1], 455./720.*img_size[0]], [1120./1280.*img_size[1], 720./720.*img_size[0]], [190. /1280.*img_size[1], 720./720.*img_size[0]]])
    dst = np.float32( [[300. /1280.*img_size[1], 100./720.*img_size[0]], [1000./1280.*img_size[1], 100./720.*img_size[0]], [1000./1280.*img_size[1], 720./720.*img_size[0]], [300. /1280.*img_size[1], 720./720.*img_size[0]]])
    """    
    # Ramesh's
    src[0] = (imgW//2 - 110, int(imgH*0.655)) # top-left (WxH)
    src[1] = (imgW//2 + 110, int(imgH*0.655)) # top-right
    src[2] = (int(imgW*0.93), int(imgH*0.96)) # botom-right
    src[3] = (int(imgW*0.07), int(imgH*0.96)) # bottom-left
    src.reshape(-1, 1, 2)
    
    offset = imgW * 0.25

    dst[0] = (0+offset, 0)
    dst[1] = (imgW-offset, 0)
    dst[2] = (imgW-offset, imgH)
    dst[3] = (0+offset, imgH)
    dst = dst.reshape((-1, 1, 2))
    
    return src, dst
    
def get_roi_coordinates(image, imgtype):
    if imgtype == "clip":
        fWidth = image.size[0]
        fHeight = image.size[1]
    else:
        fWidth = image.shape[1]
        fHeight = image.shape[0]

    top_left = [int(fWidth*0.4), int(fHeight*0.6)]
    bot_left = [int(fWidth*0.1), int(fHeight)]
    bot_right = [int(fWidth*0.9), int(fHeight)]
    top_right = [int(fWidth*0.6), int(fHeight*0.6)]
    return top_left, bot_left, bot_right, top_right

def resize_frame(frm):
    return cv2.resize(frm, (0,0), fx=0.5, fy=0.5)
        
def process_image(img):
    
    img = cv2.undistort(img, mtx, dist, None, mtx)
    gauss_img = gaussian_blur(img, 5)

    preprocessImg = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(gauss_img, orient='x', thresh=(50, 255))
    grady = abs_sobel_thresh(gauss_img, orient='y', thresh=(30, 255))
    c_binary = color_threshold(img, sthresh=(100, 255), vthresh=(60, 255))
    mag_binary = mag_thresh(gauss_img, sobel_kernel=9, mag_thresh=(50, 200))
    preprocessImg[((gradx == 1) & (grady == 1) | (c_binary == 1) | (mag_binary == 1))] = 255
    #cv2.imshow('preprocessImg', resize_frame(preprocessImg))
    
    src, dst = get_src_dst_points(img)
    #cv2.polylines(img, np.int32([src]), True, (255,0,0), 3)
    #cv2.imshow('orig', resize_frame(img))
    mask = np.zeros_like(preprocessImg)
    masked = np.array(cv2.merge((mask, mask, mask)), np.uint8)
    cv2.fillPoly(masked, np.int32([src]), color=(255, 255, 255))
    img2gray = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(preprocessImg, preprocessImg, mask=mask)
    
    preprocessImg = masked
    # Perform the transformation
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(preprocessImg, M, img_size, flags=cv2.INTER_LINEAR)
    
    window_centroids = curve_centers.find_window_centroids(warped)
    if cv2.waitKey(1) & 0xFF == 32:
        print('centroids len:', len(window_centroids))
        print('centroids', window_centroids)
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
    
    leftx = []
    rightx = []

    # Go through each level and draw the windows 	
    for level in range(0,len(window_centroids)):
        # add center value found in frame to the list of lane points per left, right
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channle 
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
    #write_name = './output_images/output_' + str(idx) + '.jpg'
    #cv2.imshow('lane masks', resize_frame(output))
    
    # fit the lane boundaries to the left, right center positions found
    yvals = range(0, warped.shape[0])
    
    res_yvals = np.arange(warped.shape[0]-(window_height/2),0, -window_height)
    
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)
    
    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)
    
    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2), axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2), axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2, right_fitx[::-1]-window_width/2), axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    
    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road,[left_lane],color=[255,0,0])
    cv2.fillPoly(road,[right_lane],color=[0,0,255])
    cv2.fillPoly(road,[inner_lane],color=[0,255,0])
    cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
    cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])
    
    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)
    
    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    output = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)
    
    ym_per_pix = curve_centers.ym
    xm_per_pix = curve_centers.xm
    
    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx, np.float32)*xm_per_pix, 2)
    curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) /np.absolute(2*curve_fit_cr[0])
    
    # calculate the offset of car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center - warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'
    
    # draw the text showing curvature, offset and speed
    cv2.putText(output,'Radius of Curvature = ' +str(round(curverad,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(output, 'Vehicle is ' +str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    
    #output = img
    return output

Output_video = 'output1_video.mp4'
Input_video = 'project_video.mp4'

window_width = 25
window_height = 80
    
# Set up the overall class to do all the tracking
curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = 15, My_ym = 9/720, My_xm = 4/384, Mysmooth_factor = 25)

clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image) # This function expects color images
video_clip.write_videofile(Output_video, audio=False)
"""

cap = cv2.VideoCapture(Input_video)
while(cap.isOpened()):
    ret, frame = cap.read()
    
    final = process_image(frame)
    cv2.imshow('final', final)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
"""