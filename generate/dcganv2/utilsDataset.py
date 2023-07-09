import numpy as np
import glob
from PIL import Image

def normImage(imageMat, dataType=np.float16):
    # Normalize the image to -1 and 1
    normImageMat = (imageMat - 127.5) / 127.5
    normImageMat = normImageMat.astype(dataType)
    return normImageMat

def cropToSquareFromEdges(images, reducedPixels, mode='maxMultipleOf'):
    # images is a matrix of shape (numImages, width, height, numChannels)
    assert(len(images.shape) == 3 or len(images.shape) == 4)
    # mode is either:
    #   'maxMultipleOf' == minimal cropping so that the resulting square images will have row and width as multiples of the reducedPixels
    #   'exact' == exactly crop the images to (reducedPixels, reducedPixels)
    assert(mode=='maxMultipleOf' or mode=='exact')
    assert(images.shape[1]>=reducedPixels and images.shape[2]>=reducedPixels)

    img_shape = images[0].shape
    if mode == 'maxMultipleOf':
        crop_pixels_x = img_shape[0] - (int(min(img_shape[0], img_shape[1])/reducedPixels)*reducedPixels)
        crop_pixels_y = img_shape[1] - (int(min(img_shape[0], img_shape[1])/reducedPixels)*reducedPixels)
    else:
        crop_pixels_x = img_shape[0] - reducedPixels
        crop_pixels_y = img_shape[1] - reducedPixels
    crop_pixels_top = int(crop_pixels_x/2)
    crop_pixels_bottom = crop_pixels_x - crop_pixels_top
    crop_pixels_left = int(crop_pixels_y/2)
    crop_pixels_right = crop_pixels_y - crop_pixels_left

    if len(images.shape) == 3:
        if not crop_pixels_x == 0:
            images = images[:, crop_pixels_top:-crop_pixels_bottom, :]
        if not crop_pixels_y == 0:
            images = images[:, :, crop_pixels_left:-crop_pixels_right]
    else:
        if not crop_pixels_x == 0:
            images = images[:, crop_pixels_top:-crop_pixels_bottom, :, :]
        if not crop_pixels_y == 0:
            images = images[:, :, crop_pixels_left:-crop_pixels_right, :]
    
    return images


def loadSegDataset(dataDir, IMAGE_SIZE, datasetName='HYTA', dataType=np.float16):
    if datasetName=='HYTA':
        image_filelist = glob.glob(dataDir + 'images/*.jpg')
        image_filelist = sorted(image_filelist)
        gtMaps_filelist = glob.glob(dataDir + '2GT/*.jpg')
        gtMaps_filelist = sorted(gtMaps_filelist)
    elif datasetName=='SWIMSEG':
        image_filelist = glob.glob(dataDir + 'images/*.png')
        image_filelist = sorted(image_filelist)
        gtMaps_filelist = glob.glob(dataDir + 'GTmaps/*.png')
        gtMaps_filelist = sorted(gtMaps_filelist)
    elif datasetName=='WSISEG':
        image_filelist = glob.glob(dataDir + 'whole-sky-images/*')
        image_filelist = sorted(image_filelist)
        gtMaps_filelist = glob.glob(dataDir + 'annotation/*')
        gtMaps_filelist = sorted(gtMaps_filelist)
    else:
        raise Exception("Only supported values for 'datasetName' are 'HYTA' and 'SWIMSEG'. Please check the input!")

    images = np.array([np.array(Image.open(fname).resize((IMAGE_SIZE,IMAGE_SIZE))) for fname in image_filelist]).astype(dataType)

    gtMaps = np.array([np.array(Image.fromarray(np.where(np.array(Image.open(fname)) < 127, 0, 255).astype(np.uint8)).resize((IMAGE_SIZE,IMAGE_SIZE)))
                  for fname in gtMaps_filelist]).astype(dataType)

    del(image_filelist, gtMaps_filelist)

    gtMaps[gtMaps<127] = 0
    gtMaps[gtMaps>=127] = 1
    gtMaps = np.expand_dims(gtMaps, axis=-1)

    return images, gtMaps

def augmentImagesFromDir(images, IMAGE_SIZE, augmentationDirPath, imageSearchStr='*.png', mode='Same', dataType=np.float16, maxNumToAug=None):
    # images is a matrix of shape (numImages, width, height, numChannels)
    # mode is either:
    #   'TreatAsBinGTmaps' == Ensure single channel and converts pixel values to binary
    #   'Same' == No edits to the read image
    assert(mode=='TreatAsBinGTmaps' or mode=='Same')

    augImage_filelist = glob.glob(augmentationDirPath + imageSearchStr)
    augImage_filelist = sorted(augImage_filelist)

    if maxNumToAug is not None:
        if len(augImage_filelist) > maxNumToAug:
            augImage_filelist = augImage_filelist[:maxNumToAug]

    if mode=='Same':
        augImages = np.array([np.array(Image.open(fname).resize((IMAGE_SIZE,IMAGE_SIZE))) for fname in augImage_filelist]).astype(dataType)
        assert(np.min(augImages)>=0 and np.max(augImages)<=255)
        # Check if the original images were already normalized
        if np.min(images)>=-1 and np.max(images)<=1:
            augImages = normImage(augImages, dataType=dataType)
        else:
            assert(np.min(images)>=0 and np.max(images)<=255)
    else:
        augImages = np.array([np.array(Image.fromarray(np.where(np.array(Image.open(fname)) < 127, 0, 255).astype(np.uint8)).resize((IMAGE_SIZE,IMAGE_SIZE)))
                        for fname in augImage_filelist]).astype(dataType)
        assert(np.min(augImages)>=0 and np.max(augImages)<=255)
        assert(len(augImages.shape)==3)
        augImages[augImages<127] = 0
        augImages[augImages>=127] = 1
        augImages = np.expand_dims(augImages, axis=-1)
    
    return np.concatenate((images, augImages), axis=0)

if __name__ == '__main__':
    images, gtMaps = loadSegDataset('../../datasets/hyta/', IMAGE_SIZE=70, datasetName='HYTA', dataType=np.float32)
    print(images.shape, gtMaps.shape)

    # images, gtMaps = loadSegDataset('../../datasets/swimseg/', IMAGE_SIZE=70, datasetName='SWIMSEG')
    # print(images.shape, gtMaps.shape)

    normalizedImages = normImage(images, dataType=np.float16)
    print(np.min(images), np.max(images), images.dtype, np.min(normalizedImages), np.max(normalizedImages), normalizedImages.dtype)

    fin_images = cropToSquareFromEdges(normalizedImages, 64, mode='exact')
    fin_gtMaps = cropToSquareFromEdges(gtMaps, 64, mode='exact')
    print(fin_images.shape, fin_gtMaps.shape)

    fin_images = cropToSquareFromEdges(normalizedImages, 16, mode='maxMultipleOf')
    fin_gtMaps = cropToSquareFromEdges(gtMaps, 16, mode='maxMultipleOf')
    print(fin_images.shape, fin_gtMaps.shape)

    aug_images = augmentImagesFromDir(fin_images, 64, '../../GAN_cloudSeg/dcgan_pix2pix_genImages/hytaAug/',
                                      imageSearchStr='*_im.png', mode='Same', dataType=np.float16, maxNumToAug=100)
    aug_gtMaps = augmentImagesFromDir(fin_gtMaps, 64, '../../GAN_cloudSeg/dcgan_pix2pix_genImages/hytaAug/',
                                      imageSearchStr='*_gt.png', mode='TreatAsBinGTmaps', dataType=np.float16, maxNumToAug=100)
    print(aug_images.shape, aug_gtMaps.shape)
    print(np.min(aug_images), np.max(aug_images), aug_images.dtype, np.min(aug_gtMaps), np.max(aug_gtMaps), aug_gtMaps.dtype)