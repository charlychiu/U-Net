#%%
import cv2

#%%
def split_image2_4patch(path2image, image_name, path2store):
    original_image = cv2.imread(path2image)
    cv2.imwrite(path2store + '/' + image_name.replace(".", "-1.") , original_image[0:572, 0:572])
    cv2.imwrite(path2store + '/' + image_name.replace(".", "-2.") , original_image[0:572, 124:696])
    cv2.imwrite(path2store + '/' + image_name.replace(".", "-3.") , original_image[124:696, 0:572])
    cv2.imwrite(path2store + '/' + image_name.replace(".", "-4.") , original_image[124:696, 124:696])

#%%
import os
file_list = list()
for dirPath, dirNames, fileNames in os.walk("./data/data1/aug"):
    file_list = fileNames

for fileNames in file_list:
    split_image2_4patch('./data/data1/aug/' + fileNames, fileNames, './data/data1/aug-patch')
    # print(fileNames)
