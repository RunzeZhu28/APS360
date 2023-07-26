import os
import shutil

def copy_all_files(src_dir, dst_dir):
    for dirpath, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            dst_file = os.path.join(dst_dir, filename)
            shutil.copy2(src_file, dst_file)

src_dir = 'C:/Users/ASUS/Desktop/musicGPT/adl-piano-midi/Rock'
dst_dir = 'C:/Users/ASUS/Desktop/musicGPT/rock'
copy_all_files(src_dir, dst_dir)
