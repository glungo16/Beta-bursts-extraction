from os.path import isfile, join
import os
import numpy as np




# Creates an csv file with all the file names within a directory
path = r"C:\Cristi\Grad McGill\Project\Beta bursts\data_raw\stroke"

files = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]

files.sort(key=lambda x: os.path.getmtime(x))

files = [f.split("\\")[-1] for f in files]
np.array(files)


exportFile = r"C:\Cristi\Grad McGill\Project\Beta bursts\data_extracted\bad_channels\bad_channels2.csv"
np.savetxt(exportFile, files, delimiter=",", fmt='%s')


# To complete the bad_channels file, open the export file in Excel and manually add all the bad channels on the second column next to the corresponding file