import glob
import os

folder = r'Y:\Users\Jie\CerCyt\BD_cytology_25\tile_patches\18XS00065'

file_paths = glob.glob(folder + '/*.tif')

for file_path in file_paths:
    _, file = os.path.split(file_path)
    file = file[:-4].replace('.ndpi_', '_')
    file = file.replace('.', '_')
    file = file+'.tif'
    path = os.path.join(folder, file)
    os.rename(file_path, path)
