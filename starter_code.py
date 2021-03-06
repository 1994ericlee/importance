#%% LOAD THE LIBRARIES
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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
data = pd.read_excel("database/teng.xlsx", usecols="B:V")
data.head(20)




#%% LOOP THROUGH THE IMAGES AND COMPUTE FEATURES
# (this code shows only size and location features; you need
# to write your own code to compute more features)


def calculate_rms(start_r, start_c, b_size, src_image):
  block = np.zeros((b_size, b_size))
  for r_step in range(b_size):
    for c_step in range(b_size):
      red = src_image[start_r+r_step][start_c+c_step][0]
      green = src_image[start_r+r_step][start_c+c_step][1]
      blue = src_image[start_r+r_step][start_c+c_step][2]
      luminance = 0.2126*red + 0.7152*green + 0.0722*blue 
      block[r_step][c_step] = luminance
      flag_matrix[start_r+r_step][start_c+c_step]=1
  contrast = block.std()/block.mean()
  
  return contrast

def dfs(sr,sc,b_size,src_image):
  for b_r in range(b_size):
    for b_c in range(b_size):
      if (sr+b_r+1 > src_img.shape[0] or sc+b_c+1 > src_img.shape[1]):
        return
      if(class_matrix[sr+b_r][sc+b_c] != obj_idx or flag_matrix[sr+b_r][sc+b_c] == 1):
        return
  contrast_array.append(calculate_rms(sr, sc, b_size, src_image))
  

   
def combine(contrast_array):
  # if (len(contrast_array)==0):
  #   return 0
  square = np.square(np.array(contrast_array))
  sum_contrast = np.sum(square)
  
  if len(contrast_array)==0:
    return 0
  a_contrast = 50/len(contrast_array)*np.sqrt(sum_contrast)
  return a_contrast

def  divide_Block(obj_idx, b_size, src_img):
  for sr in range(num_rows):
      for sc in range(num_cols):
        dfs(sr,sc,b_size,src_img)
  
  return  combine(contrast_array)      
   
#%%
# lists to hold the feature values
tot_num_objs = 581
obj_sizes = np.zeros(tot_num_objs)
obj_locs = np.zeros(tot_num_objs)
obj_contrasts = np.zeros(tot_num_objs)
obj_edges = np.zeros(tot_num_objs)
obj_fores = np.zeros(tot_num_objs)
obj_sifts = np.zeros(tot_num_objs)
idx = 0

# for each image...
# (show a progressbar using tqdm because computing features 
# can take a long time)
for img_idx in tqdm(range(1, 151)):
  img_name = "img" + str(img_idx)
  
  
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
  
  edge = cv.Canny(src_img, 100, 200)
  edge_matrix = np.array(edge)
  
  sift = cv.xfeatures2d.SIFT_create()
  k = sift.detect(src_img, None)
  pts = [p.pt for p in k]
  sift_matrix = np.array(pts).astype(int)
  sift_matrix = np.unique(sift_matrix, axis=0)
  sift_total = sift_matrix.shape[0]
  sift_flag = np.zeros(shape=(num_rows, num_cols))
  for i in range(len(sift_matrix)):
    sift_flag[sift_matrix[i][1]][sift_matrix[i][0]] = 1
  
  flag_matrix = np.zeros((num_rows, num_cols))
  class_matrix = np.ones((num_rows, num_cols))*-1
  img_B_size = []
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
    obj_edge_size = 0
    obj_in_3pix =0
    obj_total_3pix =0
    obj_sift_size = 0
    
    for r in range(num_rows):
      for c in range(num_cols):
        if msk_img[r, c, 0] == msk_r and \
          msk_img[r, c, 1] == msk_g and \
          msk_img[r, c, 2] == msk_b:
          avg_r += r
          avg_c += c
          obj_size += 1          
          
          class_matrix[r][c] = obj_idx
          
          if edge_matrix[r, c] == 255:   
            obj_edge_size += 1
            
          if r > 3 or r < num_rows-3:
            if c < 3 or c > num_cols-3:
               obj_in_3pix += 1
               
          if sift_flag[r, c] == 1:
             obj_sift_size+=1
        if r > 3 or r < num_rows-3:
            if c < 3 or c > num_cols-3:
                obj_total_3pix += 1     
          
          
          
          
    # compute the center (r,c) of the object
    avg_r /= obj_size
    avg_c /= obj_size
    
  
   
    
    # loc is the scaled avg. distance from the center
    # (actually, the scaling doesn't matter)
    obj_loc = \
      np.sqrt((avg_r-num_rows/2)**2 + (avg_c-num_cols/2)**2)
    obj_loc = 100*obj_loc/289.13175

    B_size = max(4, int(0.05*np.sqrt(obj_size)+0.05))
    img_B_size.append(B_size)
    
    
    # obj_contrast =divide_Block(obj_idx, B_size, src_img)
    # obj_contrast  = divide(B_size)
    
    obj_edge_size_percent = obj_edge_size / obj_size  
    
    obj_fore = 2-2*obj_in_3pix/obj_total_3pix
    
    obj_sift_per = obj_sift_size/sift_total
    
    # print("obj_size =", obj_size)
    # print("obj_loc =", obj_loc)
    # print("")

    # store the computed features in the lists
    obj_sizes[idx] = obj_size
    obj_locs[idx] = obj_loc
    obj_contrasts[idx] = obj_contrast
    obj_edges[idx] = obj_edge_size_percent
    obj_fores[idx] = obj_fore
    obj_sifts[idx] = obj_sift_per
    
    
    idx += 1





#%% UPDATE THE DATAFRAME WITH THE NEW FEATURES

# insert the feature lists into the dataframe
data.insert(data.shape[1]-1, "size", obj_sizes)
data.insert(data.shape[1]-1, "loc", obj_locs)
data.insert(data.shape[1]-1, "contrast", obj_contrasts)
data.insert(data.shape[1]-1, "edge", obj_edges*10)
data.insert(data.shape[1]-1, "fore", obj_fores*10)
data.insert(data.shape[1]-1, "sift", obj_sifts)
data.head(20)

# you can save to an Excel file to check the values
data.to_excel("database/tmp.xlsx")


#%% SPLIT INTO TRAINING AND TESTING

data_trn = data.iloc[:302] # first 75 images
data_tst = data.iloc[302:] # last 75 images

# in this demo, we will use only two features
# (you can use more after you've computed them)
feature_names = ["FB"]
# feature_names = ["size","loc","FB","dis_L","dis_AB","cat","importance","points","lines"]
X_trn = np.array(data_trn[feature_names])
y_trn = np.array(data_trn["class"])

X_tst = np.array(data_tst[feature_names])
y_tst = np.array(data_tst["class"])


#%% BAYES CLASSIFICATION

model = GaussianNB()
# model = SVC(kernel='linear', C=2)
# model = DecisionTreeClassifier()
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

