import os
import pandas as pd
import shutil
import re

image_dir = '/home/minsun/Downloads/PAD-UFES-20/images/imgs_part_1'
metadata_file = '/home/minsun/Downloads/PAD-UFES-20/metadata.csv'
output_base_dir = '/home/minsun/Downloads/PAD-UFES-20/images'

df = pd.read_csv(
    metadata_file,
    usecols=[0, 1, 17],
    skiprows=1,
    names=['p_id_str', 'l_id', 'diagnostic'],
    dtype={'p_id_str': str, 'l_id': str} 
)

df['p_id'] = df['p_id_str'].str.replace('PAT_', '', regex=False).astype(int)

df['l_id'] = df['l_id'].astype(int)


id_to_label = df.set_index(['p_id', 'l_id'])['diagnostic'].to_dict()

unique_labels = df['diagnostic'].unique()

for label in unique_labels:
    label_dir = os.path.join(output_base_dir, label)
    os.makedirs(label_dir, exist_ok=True)

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        match = re.match(r'PAT_(\d+)_(\d+)_\d+\.png', filename)
        if match:
            patient_id = int(match.group(1))
            lesion_id = int(match.group(2))
            lookup_key = (patient_id, lesion_id)
            if lookup_key in id_to_label:
                label = id_to_label[lookup_key]
                source_path = os.path.join(image_dir, filename)
                dest_dir = os.path.join(output_base_dir, label)
                dest_path = os.path.join(dest_dir, filename)
                shutil.move(source_path, dest_path)