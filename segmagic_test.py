from segmagic_ml import Segmagic
import tifffile as tiff
from scipy.spatial import ConvexHull
import cv2
import numpy as np
import matplotlib.pyplot as plt

# in base_path, the model(s) to use should be in a folder called "model"
# names of labels need to be defined, you can give it any name you want. Should be the same length as the number of labels in the model.
base_path = r"\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\opg paper\Segmagic"
seg = Segmagic(base_path)
labels = ['foot']
#chose image to predict
folder = r"\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\opg paper\Segmagic\foot_images"
#seg.predict_folder(folder,labels,show=True)


image_dir = r"\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\opg paper\Segmagic\foot_images\test_foot.tif"
n_clusters = 2
#def return_ratios(image_dir, n_clusters = 2):

image_to_predict = tiff.imread(image_dir).transpose(2, 0, 1)
predicted_mask, uncertainty = seg.predict_image(image_to_predict, labels, show=True)

Z = predicted_mask.reshape((-1,3))

X,Y = np.where(predicted_mask[:,:,0]==255)

Z = np.column_stack((X,Y)).astype(np.float32)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
color = label.copy().astype(str).reshape(-1)
color[color=='0'] = 'indianred'
color[color=='1'] = 'steelblue'
plt.scatter(Y,predicted_mask.shape[0]-X,color = color)

# Mark and display cluster centres 
plt.imshow(predicted_mask)
plt.plot(center[0,1],center[0,0],'o')
plt.plot(center[1,1],center[1,0],'o')
for x,y in center:
    print(f'Cluster centre: [{int(x)},{int(y)}]')
    cv2.drawMarker(predicted_mask, (int(x), int(y)), [0,50,255])





contours, hierarchy = cv2.findContours(predicted_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

hulls = []
for contour in contours:
    contour = contour.reshape(contour.shape[0],contour.shape[2])
    if contour.shape[0]>2:
        hulls.append(ConvexHull(contour))

hulls[0].vertices
len(contours)