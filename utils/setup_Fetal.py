import os
import shutil
import re

image_dir = '/home/minsun/Downloads/Fetal-Planes-DB/Images' 
output_base_dir = '/home/minsun/Downloads/Fetal-Planes-DB' 

pattern = re.compile(r'Patient\d+_(Plane\d+)_\d+_of_\d+\.png')

for filename in os.listdir(image_dir):
    match = pattern.match(filename)
    if match:
        label = match.group(1) 
        label_dir = os.path.join(output_base_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        source_path = os.path.join(image_dir, filename)
        dest_path = os.path.join(label_dir, filename)
        shutil.move(source_path, dest_path)