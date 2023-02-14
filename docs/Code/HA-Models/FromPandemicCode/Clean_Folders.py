import os
import sys


dirs_to_delete = ['./Figures/CRRA1/','./Figures/CRRA1_PVSame/',
                  './Figures/CRRA3/','./Figures/CRRA3_PVSame/',
                  './Figures/Rfree_1005/','./Figures/Rfree_1005_PVSame/',
                  './Figures/Rfree_1015/','./Figures/Rfree_1015_PVSame/',
                  './Figures/Rspell_4/','./Figures/Rspell_4_PVSame/',
                  './Figures/ADElas/','./Figures/ADElas_PVSame/',
                  './Figures/LowerUBnoB/','./Figures/LowerUBnoB_PVSame/']

for dir_to_delete in dirs_to_delete:
    print('To delete: ', dir_to_delete)
    All_Files = os.listdir(dir_to_delete)
    
    
    for item in All_Files:
        if os.path.getsize(os.path.join(dir_to_delete,item)) > 1000000: #larger than 1MB
            os.remove(os.path.join(dir_to_delete,item))