from __future__ import print_function
import cv2
import numpy as np
import rawpy
import glob
import os

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15





def alignImages(im1, im2):

    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h





def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)

    im = np.maximum(im - 64, 0)/ (1023 - 64) #subtract the black level

    im = np.expand_dims(im,axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:],
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out



input_dir = './dataset/seattle_night_raw_only/'
gt_dir='./dataset/seattle_night_gt_only/'
train_fns = glob.glob(gt_dir + '0*.jpg')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append((train_fns[i],int(train_fn[0:4])))

# test_fns = glob.glob(gt_dir + '0*.dng')
# test_ids = []
# for i in range(len(test_fns)):
#     _, test_fn = os.path.split(test_fns[i])
#     test_ids.append((test_fns[i],int(test_fn[0:4])))

# print(train_fns)
input_fns = glob.glob(input_dir + '0*.dng')
input_ids=[]
for i in range(len(input_fns)):
    _, input_fn = os.path.split(input_fns[i])
    input_ids.append((input_fns[i],int(input_fn[0:4])))


print(len(input_ids))
print(len(train_ids))
input_dic={}
for item in input_ids:
    input_dic[str(item[1])]=item[0]

for item in train_fns:
    refFilename = "0000_out.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    print(imReference.shape)


    imRaw='0000.dng'
    raw=rawpy.imread(imRaw)


    raw=raw.postprocess(use_camera_wb=False, half_size=False, no_auto_bright=True)
    raw=raw[25:3040-25, 35:4056-35]
    print(raw.shape)




    imReg, h = alignImages(imReference, raw)


    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename);

    # cv2.imwrite(outFilename, imReg[:,:,::-1])
    cv2.imwrite(outFilename, imReg)


