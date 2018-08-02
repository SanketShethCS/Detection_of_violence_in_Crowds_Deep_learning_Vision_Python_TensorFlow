'''
This script renames image files as either violent or non_violent by appending v or nv to their names respectively
'''

import os
os.getcwd()
collection = "D:\\RIT\\631\\Project\\data_project\\Violence"
for i, filename in enumerate(os.listdir(collection)):
    os.rename(filename,  str(i)+"-v" + ".jpg")