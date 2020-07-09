
import os
from shutil import copy2

if not os.path.isdir("./temp"):
    os.mkdir('./temp')

filepath = "./simulated_trinity_data/10nodes_40len_trial6.xlsx"

copy2(filepath,'./temp/')
# print(os.path.basename(filepath))
