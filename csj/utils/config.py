# coding: utf-8
import os

duts_root_test ='datasets/DUTS/DUTS-TE'
COD10K_root_test ='datasets/COD10K/TestDataset/TestDataset/COD10K'
polyp_root_test = 'datasets/unified_lesion_dataset/polyp_five_test_dataset'
SBU_root_Test ='datasets/SBU-shadow/SBU-Test_rename'
COVID_root_test = 'datasets/unified_lesion_dataset/COVID-19_Lung_Infection_test/COVID'
Breast_root_test = 'datasets/unified_lesion_dataset/breast/test'
trans10k_root_test_easy = 'datasets/Trans10K/test/easy'
trans10k_root_test_hard = 'datasets/Trans10K/test/hard'
trans10k_root_test_all = 'datasets/Trans10K/test/all'
isic2018_root_test = 'datasets/unified_lesion_dataset/isic2018/val'


duts = os.path.join(duts_root_test)
COD10K =  os.path.join(COD10K_root_test)
SBU = os.path.join(SBU_root_Test)
trans10k_easy = os.path.join(trans10k_root_test_easy)
trans10k_hard = os.path.join(trans10k_root_test_hard)
trans10k_all = os.path.join(trans10k_root_test_all)
polypfive = os.path.join(polyp_root_test)
covid = os.path.join(COVID_root_test)
breast = os.path.join(Breast_root_test)
isic2018  = os.path.join(isic2018_root_test)

