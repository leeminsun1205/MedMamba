import splitfolders
import os

input_folder = '/home/minsun/Downloads/PAD-UFES-20' 
output_folder = '/home/minsun/Downloads/PAD-UFES-20'        

splitfolders.ratio(input_folder,
                   output=output_folder,
                   seed=42,
                   ratio=(0.6, 0.1, 0.3),
                   group_prefix=None, 
                   move=False 
                   )

print(f"Dataset đã được chia vào thư mục: {os.path.abspath(output_folder)}")