import os

DIR_DATA_SCIENCE_BOWL_2018 = r'Y:\Cytology\data-science-bowl-2018'      # the Kaggle dataset root folder
DIR_NUCLEUS_MRCNN = r'Y:\Users\Jie\CerCyt\nucleus_mrcnn'                # MRCNN nucleus segmentation root folder
DIR_NUCLEUS_MRCNN_LOGS = os.path.join(DIR_NUCLEUS_MRCNN, 'logs')        # MRCNN nucleus segmentation log folder
DIR_NUCLEUS_MRCNN_MODELS = os.path.join(DIR_NUCLEUS_MRCNN, 'models')    # MRCNN nucleus segmentation model folder

DIR_NLM_DATA = r'Y:\Cytology\NLM_Data'                                  # Our dataset root folder
DIR_NLM_CLEAN2 = os.path.join(DIR_NLM_DATA, 'cLEAN2')                   # Our dataset clean NDPI files
DIR_NLM_CLEAN2_LEVEL3 = os.path.join(DIR_NLM_DATA, 'cLEAN2_Level3')     # Level 3 images of our dataset clean NDPI files
DIR_NLM_INK = os.path.join(DIR_NLM_DATA, 'ink')                         # Our dataset NDPI files with blue inks
DIR_NLM_ANNOTATED = os.path.join(DIR_NLM_DATA, 'annotated')             # Our dataset with abnormal cell annotations


DIR_MY_DATA = r'Y:\Users\Jie\CerCyt'                                    # My CerCyt project data root folder
DIR_BD_cytology_25 = os.path.join(DIR_MY_DATA, 'BD_cytology_25')        # My BD_cytology_25 data root folder
DIR_NLM_PATCH = os.path.join(DIR_BD_cytology_25, 'patches')             # The patches cropped from clean NDPI files
DIR_INK_LOW_RES = os.path.join(DIR_BD_cytology_25, 'ink_low_res')       # The low resolution ink image for viz registration to cLEAN2
DIR_BLUE_ROI = os.path.join(DIR_BD_cytology_25, 'blue_rois')            # The patches cropped from cLEAR2 under blue marks
DIR_NUCLEUS_MRCNN_RESULT = os.path.join(DIR_BD_cytology_25, 'MRCNN')    # MRCNN nucleus segmentation results
DIR_ABNORMAL_CELL_PATCH = os.path.join(DIR_BD_cytology_25, 'abnormal_cell_patches')   # The abnormal cell patches cropped from abnormal cell annotation
DIR_NORMAL_CELL_PATCH = os.path.join(DIR_BD_cytology_25, 'normal_cell_patches')   # The normal cell patches cropped from mrcnn segmentation results
