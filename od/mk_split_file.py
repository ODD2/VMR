import os
from glob import glob

files = glob("/scratch4/users/od/YTCharts/v1/test/*.mp4")
with open("/scratch4/users/od/YTCharts/splits/test.txt", "w") as f:
    for file in files:
        f.write(os.path.basename(file)[:-4] + "\n")
