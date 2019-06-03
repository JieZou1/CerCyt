# CerCyt
Cervical Cytology Project

## Cell patch based classification

Use Masked-RCNN to do nuclei instance segmentation. Then, based on the detected nuclei center, crop
a patch. This patch is called cell patch, in comparison to the previous methods based on scanning window
patch. The classification is based on the cell patches.

Steps:

### Create blue inks ROIs
All G Tom annotation is under blue ink roi, so we should crop the ROI out. 
Currently, the ROI coordinates in level 3 resolution is at Y:\Cytology\generated_out\rois_matched_level3.
However, some ROI may miss G Tom annotation, so we may need to modify some ROIs.
The finally ROIs are at Y:\Users\Jie\CerCyt\BD_cytology_25\blue_rois.

* We convert level 3 coordinates to level 0 coordinates and crop the corresponding ROI patches
* If we notice misses of G Tom's annotation, we manually increase the ROI.

### Check whether G Tom's annotation is outside ROI
Find Ink and cLEAN2 registration difference;
Read G Tom annotation on Ink slide and convert them to cLEAR2 coordinates, 
Detect any annotation outside of the current Ink ROI, and then manually increase ROI to include all G Tom Annotation.

### Use Mask-RCNN to detect all nuclei in the ROI

### Generate normal and abnormal cell patches from detected nuclei
Any nuclei inside G Tom annotation is counted as abnormal; otherwise, normal

### Train a CNN model to do normal/abnormal cell classification
Partition the 25 slides into training and evluation set



1. preprocessing/crop_into_patches.py
   Crop the original NDPI images to 1024x1024 patches with overlapping 128 pixels. <BR>
   The results are at Y:\Users\Jie\CerCyt\BD_cytology_25\patches

2. cell_roi/cellular_roi_v2.py to generate blue ink marked bounding boxes.<BR>
   The low resolution images after alignment is at Y:\Users\Jie\CerCyt\BD-cytology-25\ink_low_res. <BR>
   The blue ink bounding boxes are saved at Y:\Users\Jie\CerCyt\BD-cytology-25\blue_rois.<BR>
   It includes the normalized bounding boxes (_rois.txt files),
   all bounding boxes drawn on low resolution images (12XS00147_ink_rois.tif); and 
   the ROI patches cropped from level 0, all these are in the coordinates of cLEAN slides. 
   
3. nucleus_mrcnn/nucleus_mrcnn.py 
   Train and segment nuclei.<BR>
   We use Kaggle dataset, at Y:\Cytology\data-science-bowl-2018.<BR>
   Use all Kaggle stage1 train samples (stage1_train), the trained model is 
   Y:\Users\Jie\CerCyt\nucleus_mrcnn\models\mask_rcnn_nucleus_0040-all_kaggle.h5<BR>
   Then, we use 16 Cytology image samples (stage1_train-cervical), the trained model is  
   Y:\Users\Jie\CerCyt\nucleus_mrcnn\models\mask_rcnn_nucleus_0040-16_kaggle.h5.<BR>
   mask_rcnn_nucleus_0040-all_kaggle.h5 is the starting model for mask_rcnn_nucleus_0040-16_kaggle.h5.<BR>
   
   The segmentation result of all 1024x1024 patches is at Y:\Users\Jie\CerCyt\BD-cytology-25\patches_MCRNN_segment.
   The segmentation result of patches under blue ink bounding boxes are at Y:\Users\Jie\CerCyt\BD_cytology_25\blue_rois_mcrnn_segment

4. preprocessing/gen_abnormal_cell_patches_from_ndpa.py
   Parse NDPA manual annotation to generate abnormal cell patches.
   Results are saved at Y:\Users\Jie\CerCyt\BD_cytology_25\abnormal_cell_patches
   
5. preprocessing/gen_normal_cell_patches_from_mrcnn
   Use mrcnn to detect nuclei in blue rois (generated in step 2), and then extract 256x256 patches as normal cell patch. 
   Results are saved at Y:\Users\Jie\CerCyt\BD_cytology_25\normal_cell_patches
    
