# CerCyt/BD_cytology_25/cell_patch_classification

Cell patch based classification on 25 BD Cytology datasets 

Use Masked-RCNN to do nuclei instance segmentation. Then, based on the detected nuclei center, crop
a patch. This patch is called cell patch, in comparison to the previous methods based on scanning window
patch. The classification is based on the cell patches.

Y: drive is mapped to \\lhcdevfiler\cervixca\
Z: drive is mapped to \\ceb-na\jie_project\

Steps:

### crop_into_tile_patches
Crop cLEARN2 slide level 0 into 1024x1024 (with 128 pixel overlap) tiles, and saved at 
Y:\Users\Jie\CerCyt\BD_cytology_25\tile_patches

### ink_cLEAN2_registration
The ink and cLEAN2 slides have some translation transform.
We need to find the translation parameters, such that 
we could convert NDPA annotations, which is on ink slides, to cLEAN2 coordinates. 
It reads cLEAN2 and ink slides, and find the translation parameters, 
the results are save at Y:\Users\Jie\CerCyt\BD_cytology_25\cell_patch_classification\image_align_ink2clean. 
It is the parameters transforming ink images to cLEAN2 images. 

### crop_NDPA_annotation_from_cLEAN2
G Tom NDPA annotation is on ink slides, we read the annotation, converts them to cLEAN2 coordinates, 
crop corresponding patches from cLEAN2, and then save the patches at 
Y:\Users\Jie\CerCyt\BD_cytology_25\cell_patch_classification\G_Tom_Patch_abnormal, 
Y:\Users\Jie\CerCyt\BD_cytology_25\cell_patch_classification\G_Tom_Patch_malignancy, and 
Y:\Users\Jie\CerCyt\BD_cytology_25\cell_patch_classification\G_Tom_Patch_normal folders.
 
### mcrnn_segment_abnormal
Use Mask-RCNN to detect all nuclei in the abnormal patches. 
For patches in Y:\Users\Jie\CerCyt\BD_cytology_25\cell_patch_classification\G_Tom_Patch_abnormal, the results 
are saved at Y:\Users\Jie\CerCyt\BD_cytology_25\cell_patch_classification\cell_patch_abnormal

### mcrnn_segment_normal
Use Mask-RCNN to detect some nuclei in the normal slides. 
The results are saved at Y:\Users\Jie\CerCyt\BD_cytology_25\cell_patch_classification\cell_patch_normal

### prepare_classification
Move the images to the folders to prepare to use ImageDataGenerator for Keras training

### cell_patch_classification
To Train and Classify cell patches

### nuclei_segmentation_prepare_normal
Collect some 1024x1024 patches from normal NDPI slides for evaluate MRCNN nuclei segmentation

### nuclei_segmentation_prepare_abnormal
Collect some 1024x1024 patches from abnormal NDPI slides for evaluate MRCNN nuclei segmentation

### nuclei_segmentation_mrcnn
Do automatic MRCNN nuclei segmentation


## Below are some obsolete scripts  


### convert_roi_level3_to_level0
The original ROI coordinates in level 3 resolution is at Y:\Cytology\generated_out\rois_matched_level3.
We convert level 3 coordinates to level 0 coordinates. 
The result is saved at Y:\Users\Jie\CerCyt\BD_cytology_25\cell_patch_classification\roi_level0

### crop_roi_from_cLEAN2
From level 0 ROI coordinates, at Y:\Users\Jie\CerCyt\BD_cytology_25\cell_patch_classification\roi_level0, 
we crop the ROI region in level 0 resolution, the results are saved at: 
Y:\Users\Jie\CerCyt\BD_cytology_25\cell_patch_classification\patch_level0

### check_NDPA_and_blue_roi
Check to make sure that NDPA manual annotation is included in blue ROI. 
NDPA annotation is at Y:\Cytology\NLM_Data\annotated.  
If not, we manually increase blue ROI. 




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

### Generate normal and abnormal cell patches from detected nuclei
Any nuclei inside G Tom annotation is counted as abnormal; otherwise, normal

### Train a CNN model to do normal/abnormal cell classification
Partition the 25 slides into training and evaluation set
