import os

path = "./Cosmetic/defect/"
file_list = os.listdir(path)

for i in range(len(file_list)):
    print(file_list[i])