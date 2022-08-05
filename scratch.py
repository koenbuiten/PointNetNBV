import os

for folder in os.listdir('data/table_0444'):
    for file in os.listdir(os.path.join('data/table_0444',folder)):
        new_file = file.split('.')[0].split('_')
        new_file = "_".join(["_".join(new_file[0:2]),"_".join(new_file[3:])]) + '.' + file.split('.')[-1]
        # print(new_file)
        os.rename(os.path.join('data/table_0444',folder,file),os.path.join('data/table_0444',folder,new_file))