#%% LOAD THE LIBRARIES
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import cv2 as cv

import matplotlib
matplotlib.rcParams['figure.dpi'] = 125
import matplotlib.pyplot as plt
from imutils import imshow

from tqdm import tqdm


#%% LOAD THE DATA
data = pd.read_excel("database/img_importance.xlsx", usecols="B:H")
data.head(20)




#%% LOOP THROUGH THE IMAGES AND COMPUTE FEATURES
# (this code shows only size and location features; you need
# to write your own code to compute more features)
def check(start_r, start_c, class_matrix, obj_idx, b_size):

  for r_step in range(b_size):
    for c_step in range(b_size):
      if(start_r + b_size > class_matrix.shape[0]) or (start_c + b_size > class_matrix.shape[1]):
        return False
      if (class_matrix[start_r+r_step][start_c+c_step] != obj_idx) or (flag_matrix[start_r+r_step][start_c+c_step] == 1):
        return False
      
  return True

def calculate_rms(start_r, start_c, b_size, src_image):
  block = np.zeros((b_size, b_size))
  for r_step in range(b_size):
    for c_step in range(b_size):
      red = src_image[start_r+r_step][start_c+c_step][0]
      green = src_image[start_r+r_step][start_c+c_step][1]
      blue = src_image[start_r+r_step][start_c+c_step][2]
      luminance = 0.2126*red + 0.7152*green + 0.0722*blue 
      block[r_step][c_step] = luminance
  contrast = block.std()/block.mean()
  return contrast

# def dfs(start_r,  start_c,  b_size, src_image, contrast_array):
  
#   if (start_c < 0 or start_r < 0 or (start_r + b_size) > num_rows or (start_c + b_size) > num_cols):
#     return
  
#   if (not check(start_r, start_c, class_matrix, obj_idx, b_size)):
#     dfs(start_r, start_c + 1, b_size, src_image, contrast_array)
#     dfs(start_r, start_c - 1, b_size, src_image, contrast_array)
#     dfs(start_r+1, start_c, b_size, src_image, contrast_array)
#     dfs(start_r-1, start_c, b_size, src_image, contrast_array)
#   else:
#     print(start_r, start_c)
#     for r_step in range(b_size):
#       for c_step in range(b_size):
#         flag_matrix[start_r+r_step][start_c+c_step]=1
#     contrast_array.append(calculate_rms(start_r, start_c, b_size, src_image))
#     return len(contrast_array)  
def combine(contrast_array):
  if (len(contrast_array)==0):
    return 0
  square = np.square(np.array(contrast_array))
  sum_contrast = np.sum(square)
  a_contrast = 50/len(contrast_array)*np.sqrt(sum_contrast)
  return a_contrast

def divide(b_size):
    for sr in range(num_rows):
      for sc in range(num_cols):
        if class_matrix[sr][sc] == obj_idx and flag_matrix[sr][sc] == 0:
         
          if check(sr, sc, class_matrix, obj_idx, b_size):
            for r_step in range(b_size):
              for c_step in range(b_size):
                flag_matrix[sr+r_step][sc+c_step]=1
            contrast_array.append(calculate_rms(sr, sc, b_size, src_img))
            
    return combine(contrast_array) 

# lists to hold the feature values
tot_num_objs = 581
obj_sizes = np.zeros(tot_num_objs)
obj_locs = np.zeros(tot_num_objs)
obj_contrasts = np.zeros(tot_num_objs)
obj_m_rgbs = np.zeros(tot_num_objs)
idx = 0

# for each image...
# (show a progressbar using tqdm because computing features 
# can take a long time)
for img_idx in tqdm(range(1, 151)):
  img_name = "img" + str(img_idx)
  # print("--- Image:", img_name, " -----------")
  
  # extract the dataframe rows just for this image
  img_data = data.loc[data["img_name"]==img_name]

  msk_name = img_name + "_msk"
  src_img = cv.imread("database/imgs/" + img_name + ".bmp")
  src_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)
  msk_img = cv.imread("database/imgs/msk/" + msk_name + ".bmp")
  msk_img = cv.cvtColor(msk_img, cv.COLOR_BGR2RGB)

  num_rows = src_img.shape[0]
  num_cols = src_img.shape[1]

  # ipcv_plt.imshow(src_img, zoom=0.5)
  # ipcv_plt.imshow(msk_img, zoom=0.5)
  flag_matrix = np.zeros((num_rows, num_cols))
  class_matrix = np.ones((num_rows, num_cols))*-1
  # for each object of the image...
  for obj_idx, obj_data in img_data.iterrows():

    contrast_array = []
    obj_name = obj_data["obj_name"]
    msk_r = obj_data["R"]
    msk_g = obj_data["G"]
    msk_b = obj_data["B"]

    # print("obj_name =", obj_name)
    # print("obj_idx = ", obj_idx)
    # print("mask (R,G,B) = (", 
    #   msk_r, ",", msk_g, ",", msk_b, ")")

    # compute size and location features
    avg_r = 0
    avg_c = 0
    obj_size = 0
    sum_red = 0
    sum_green = 0
    sum_blue = 0
    for r in range(num_rows):
      for c in range(num_cols):
        if msk_img[r, c, 0] == msk_r and \
          msk_img[r, c, 1] == msk_g and \
          msk_img[r, c, 2] == msk_b:
          avg_r += r
          avg_c += c
          obj_size += 1          
          class_matrix[r][c] = obj_idx
          sum_red += src_img[r, c, 0]
          sum_green += src_img[r, c, 1]
          sum_blue += src_img[r, c, 0]
    # compute the center (r,c) of the object
    avg_r /= obj_size
    avg_c /= obj_size
    
    obj_m_rgb = (sum_red+sum_blue+sum_green)/obj_size/3
    # loc is the scaled avg. distance from the center
    # (actually, the scaling doesn't matter)
    obj_loc = \
      np.sqrt((avg_r-num_rows/2)**2 + (avg_c-num_cols/2)**2)
    obj_loc = 100*obj_loc/289.13175

    B_size = max(4, int(0.05*np.sqrt(obj_size)+0.05))
    
    
    # obj_contrast  = divide(B_size)
    
   
    
    
    # print("obj_size =", obj_size)
    # print("obj_loc =", obj_loc)
    # print("")

    # store the computed features in the lists
    obj_sizes[idx] = obj_size
    obj_locs[idx] = obj_loc
    # obj_contrasts[idx] = obj_contrast
    obj_m_rgbs[idx]= obj_m_rgb
    idx += 1




#%% UPDATE THE DATAFRAME WITH THE NEW FEATURES

# insert the feature lists into the dataframe
data.insert(data.shape[1]-1, "size", obj_sizes)
data.insert(data.shape[1]-1, "loc", obj_locs)
# data.insert(data.shape[1]-1, "contrast", obj_contrasts)
data.insert(data.shape[1]-1, "m_rgb", obj_m_rgbs)
data.head(20)

# you can save to an Excel file to check the values
#data.to_excel("imp/tmp.xlsx")


#%% SPLIT INTO TRAINING AND TESTING

data_trn = data.iloc[:302] # first 75 images
data_tst = data.iloc[302:] # last 75 images

# in this demo, we will use only two features
# (you can use more after you've computed them)
feature_names = ["size", "loc", "m_rgb"]

X_trn = np.array(data_trn[feature_names])
y_trn = np.array(data_trn["class"])

X_tst = np.array(data_tst[feature_names])
y_tst = np.array(data_tst["class"])


#%% BAYES CLASSIFICATION

model = GaussianNB()
model.fit(X_trn, y_trn)

y_trn_prd = model.predict(X_trn)
print('Training accuracy: ', 
  accuracy_score(y_true=y_trn, y_pred=y_trn_prd))

y_tst_prd = model.predict(X_tst)
print('Testing accuracy: ', 
  accuracy_score(y_true=y_tst, y_pred=y_tst_prd))

# TODO: Add classification summary table
# TODO: Add confusion matrix


#%% SHOW GT AND PREDICTED IMPORTANCE MAP

# specify the image you want to view
img_name = "img150" 
idx = data_tst.index[data_tst["img_name"]==img_name][0] - 302

img_name = data_tst.iat[idx, 1]
print("Index: ", idx)
print("Image Name:", img_name)
print()

src_img = cv.imread("database/imgs/" + img_name + ".bmp")
src_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)
msk_img = cv.imread(
  "database/imgs/msk/" + img_name + "_msk.bmp")
msk_img = cv.cvtColor(msk_img, cv.COLOR_BGR2RGB)    

print("Original image")
imshow(src_img, zoom=0.5)
print("Mask image")
imshow(msk_img, zoom=0.5)

num_rows = src_img.shape[0]
num_cols = src_img.shape[1]

imp_map = np.zeros((num_rows, num_cols))
imp_map_prd = np.zeros((num_rows, num_cols))

num_objs = int(data_tst.iat[idx, 0])
for obj_idx in range(num_objs):
  obj_name = data_tst.iat[idx+obj_idx, 2]
  obj_clr_r = data_tst.iat[idx+obj_idx, 3]
  obj_clr_g = data_tst.iat[idx+obj_idx, 4]
  obj_clr_b = data_tst.iat[idx+obj_idx, 5]

  # class_label = data_tst.iat[idx+obj_idx, 12]
  class_label = y_tst[idx+obj_idx]
  class_label_prd = model.predict((X_tst[idx+obj_idx],))
  imp_val = float(class_label) * 127.5
  imp_val_prd = float(class_label_prd) * 127.5

  for r in range(num_rows):
    for c in range(num_cols):
      if (msk_img[r, c, 0] == obj_clr_r) and \
        (msk_img[r, c, 1] == obj_clr_g) and \
        (msk_img[r, c, 2] == obj_clr_b):
        imp_map[r, c] = imp_val
        imp_map_prd[r, c] = imp_val_prd

print("Ground-truth importance map")
imshow(imp_map, cmap="gray", 
  vmin=0, vmax=255, zoom=0.5)

print("Predicted importance map")
imshow(imp_map_prd, cmap="gray", 
  vmin=0, vmax=255, zoom=0.5)

