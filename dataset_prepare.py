import os, cv2,itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split

"""
This file is used to get the balanced traning data, its labels and the validation data 
and its labels.
"""



def dataprepare():
	### Change your path here
	data_dir="/afs/crc.nd.edu/user/p/pgu/Research/CV_project/Skin_lesion_classification/train/"
	#print(os.listdir(data_dir))
	all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
	#print(all_image_path)
	imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
	#print(imageid_path_dict)
	lesion_type_dict = {
	    'nv': 'Melanocytic nevi',  # 4
	    'mel': 'dermatofibroma',  # 6
	    'bkl': 'Benign keratosis-like lesions ', # 2
	    'bcc': 'Basal cell carcinoma', # 1
	    'akiec': 'Actinic keratoses', # 0
	    'vasc': 'Vascular lesions', # 5
	    'df': 'Dermatofibroma' # 3
	}
	df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
	df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
	df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
	df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
	
	print('This is the original dataframe after adding path, cell_type and cell_type_idx', df_original)
	f1 = open('df_original.pd', 'wb')
	pickle.dump(df_original, f1)



	df_undup = df_original.groupby('lesion_id').count()
	# now we filter out lesion_id's that have only one image associated with it
	df_undup = df_undup[df_undup['image_id'] == 1]
	df_undup.reset_index(inplace=True)
	#print('This is the df_undup', df_undup)
	# here we identify lesion_id's that have duplicate images and those that have only one image.
	def get_duplicates(x):
	    unique_list = list(df_undup['lesion_id'])
	    if x in unique_list:
	        return 'unduplicated'
	    else:
	        return 'duplicated'

	# create a new colum that is a copy of the lesion_id column
	df_original['duplicates'] = df_original['lesion_id']
	# apply the function to this new column
	df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
	#print(df_original)
	


	print(df_original['duplicates'].value_counts())
	df_undup = df_original[df_original['duplicates'] == 'unduplicated']
	print(df_undup.shape)
	# now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
	y = df_undup['cell_type_idx']
	_, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
	print(df_val.shape)
	#print(df_val)
	print(df_val['cell_type_idx'].value_counts())
	# This set will be df_original excluding all rows that are in the val set
	# This function identifies if an image is part of the train or val set.
	def get_val_rows(x):
	    # create a list of all the lesion_id's in the val set
	    val_list = list(df_val['image_id'])
	    if str(x) in val_list:
	        return 'val'
	    else:
	        return 'train'

	# identify train and val rows
	# create a new colum that is a copy of the image_id column
	df_original['train_or_val'] = df_original['image_id']
	# apply the function to this new column
	df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
	# filter out train rows
	df_train = df_original[df_original['train_or_val'] == 'train']
	###### The df_train and df_val before balanced 
	print(len(df_train))
	print(len(df_val))



	##### Do balanced  here for df_train and df_val 
	print('This is the df_train', df_train)
	print(df_train['cell_type_idx'].value_counts())
	print(df_val['cell_type'].value_counts())
	print('This is the df_val', df_val)
	# Copy fewer class to balance the number of 7 classes
	data_aug_rate = [15,10,5,50,0,40,5]
	for i in range(7):
	    if data_aug_rate[i]:
	        df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)
	print(df_train['cell_type'].value_counts())
	print(df_train)
	df_train = df_train.reset_index()
	print('This is the df_train after balanced',df_train)
	df_val = df_val.reset_index()
	print('This is the df_val after balanced', df_val)
	#### save the labels csv files
	# df_train.to_csv (r'/afs/crc.nd.edu/user/p/pgu/Research/CV_project/Skin_lesion_classification/train/train_label_balanced.csv', index = None, header=True)
	# df_val.to_csv (r'/afs/crc.nd.edu/user/p/pgu/Research/CV_project/Skin_lesion_classification/train/validation_label.csv', index = None, header=True)

	f = open('df_train.pd', 'wb')
	pickle.dump(df_train, f)

	f = open('df_val.pd', 'wb')
	pickle.dump(df_val, f)


if __name__ == '__main__':
    dataprepare()