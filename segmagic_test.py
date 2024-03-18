from segmagic_ml import Segmagic
import tifffile as tiff
from scipy.spatial import ConvexHull
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops


import pandas as pd
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap,QFont,QImage
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import os

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def find_intensity(image_dir,data):
    seg = Segmagic(base_path)
    name_split = image_dir.replace(".tif","").split("/")[-1].split("\\")[-1].split("_")
    if len(name_split)!=3:
        raise Warning(f"Make sure to use the following naming scheme 'subjectid_extremity_phase#' (e.g. CRPS001_hand_3).\n\
                    The given name is {'_'.join(name_split)}")
    # Extract importat information
    id, model_type, phase = name_split
    afctd_side = data.loc[data['ID']==id,'affected_extremity'].values[0]
    labels = [model_type]

    # predict mask
    image_to_predict = tiff.imread(image_dir).transpose(2, 0, 1)
    predicted_mask, uncertainty = seg.predict_image(image_to_predict, labels, show=True)
    labeled_mask, n_masks = label(predicted_mask, return_num=True)
    
    # if the resulting mask is just one big mask, add a vertical line to split the image into two
    if n_masks==1:
        print("\nOne big region was found. Splitting using a vertical line\n")
        image_to_predict[:,:,190:210] = 255
        predicted_mask, uncertainty = seg.predict_image(image_to_predict, labels, show=True)
        labeled_mask, n_masks = label(predicted_mask, return_num=True)
    # Extract largest regions
    region_inds, region_count = np.unique(labeled_mask, return_counts=True)
    largest_regions = np.argsort(region_count)[::-1][1:3]

    # choose the biggest 2 regions and label them
    filtered_mask = np.where(~np.isin(labeled_mask,largest_regions),0,labeled_mask)
    filtered_regions = label(filtered_mask)
    img = image_to_predict.transpose((1,2,0))
    props = regionprops(filtered_regions[:,:,0],img)


    # Extract centroids and side of affected extremity
    centroids = [region.centroid for region in props]
    region_side = ["right" if centroid[1]>200 else "left" for centroid in centroids]
    region_afctd_extr = ["ipsi" if side==afctd_side else "contra" for side in region_side]

    # compute intensities
    intensities = [255-np.mean(prop.intensity_mean) for prop in props]
    intensity_dic = {side:intensity for side,intensity in zip(region_afctd_extr,intensities)}
    if len(intensity_dic.keys()) ==1:
        print ("Segmentation Failed")
        return None, None, None, None, None
    intensity_dic['ratio'] = intensity_dic['ipsi']/intensity_dic['contra']

    return image_to_predict,filtered_mask, centroids, region_afctd_extr, intensity_dic

# Choose by image
base_path = r"\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\opg paper\Segmagic"
data = pd.read_excel(base_path+'/pt_info.xlsx')
#image_dir = r"\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\opg paper\Segmagic\all_images\CRPS004_hand_3.tif" #CRPS007P3#CRPS004P1

#find_intensity(image_dir,data, show = True)


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Create main layout
        main_layout = QVBoxLayout()

        # Create QHBoxLayout for labels
        hbox_layout = QHBoxLayout()
        
        # Create QLabel objects
        self.title_label = QLabel("ScintiSegmenter (or ScintIntensify)", self)
        font_title = QFont("Calibri", 18)
        self.title_label.setFont(font_title)
        self.title_label.setAlignment(Qt.AlignCenter)  # Align text to the center


        sub_text = "Choose the image to be segmented and intensity calculated.\nThe image file should have the following naming scheme 'subjectid_extremity_phase#' (e.g. CRPS001_hand_3)"
        self.sub_label = QLabel(sub_text, self)
        font_sub = QFont("Calibri", 12)
        self.sub_label.setFont(font_sub)
        self.sub_label.setAlignment(Qt.AlignCenter)  # Align text to the center

        self.label1 = QLabel()
        self.label2 = QLabel()

        self.label1.setFixedSize(420, 400)
        self.label2.setFixedSize(420, 400)

        # Add QLabel objects to QHBoxLayout
        hbox_layout.addWidget(self.label1)
        hbox_layout.addWidget(self.label2)
        
        # Create QLineEdit and QPushButton
        self.image_path_line = QLineEdit()
        browse_button = QPushButton("Load Image")
        segment_button = QPushButton("Segment")
        exit_button = QPushButton("Exit")

        # Connect button click event to function
        browse_button.clicked.connect(self.browse_image)
        segment_button.clicked.connect(self.segment)
        exit_button.clicked.connect(self.close)
        
        # Add QHBoxLayout, button, and QLineEdit to QVBoxLayout
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.sub_label)
        main_layout.addLayout(hbox_layout)
        main_layout.addWidget(self.image_path_line)
        main_layout.addWidget(browse_button)
        main_layout.addWidget(segment_button)
        main_layout.addWidget(exit_button)

        # Set the layout
        self.setLayout(main_layout)

    def browse_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.tif);;All Files (*)", options=options)
        if file_path:
            self.image_path_line.setText(file_path)
            self.load_image()


    def load_image(self):
        image_path = self.image_path_line.text()
        pixmap = QPixmap(image_path)
        # Set the pixmap to the first QLabel
        self.label1.setPixmap(pixmap.scaled(self.label1.size(), QtCore.Qt.KeepAspectRatio))
        self.label2.clear()
    def segment(self):
        image_dir = self.image_path_line.text()
        if os.path.exists(image_dir):
            image_to_predict,filtered_mask, centroids, region_afctd_extr, intensity_dic = find_intensity(image_dir,data)
            color_list = ['steelblue', 'indianred','olivedrab','darkgoldenrod','darkmagenta','grey','palevioletred','sienna','beige','coral']

            fig = plt.subplots(figsize = (5,5))
            cmap = plt.cm.colors.ListedColormap(['white']+color_list[0:len(centroids)])
            plt.imshow(image_to_predict[0,:,:],cmap='gray')
            plt.imshow(filtered_mask, cmap=cmap, interpolation='nearest', alpha = 0.3)
            # Add text
            for centroid,l_text in zip(centroids,region_afctd_extr):
                plt.text(centroid[1],centroid[0],l_text, ha='center', font = 'Calibri', size = 20)

            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            plt.savefig(base_path+"/temp_file.png")
            
            pixmap_regions = QPixmap(base_path+"/temp_file.png")
            self.label2.setPixmap(pixmap_regions.scaled(self.label2.size(), QtCore.Qt.KeepAspectRatio))

    def fig_to_pixmap(self, fig):
        """Convert matplotlib figure to QPixmap."""
        # Render the figure to a QImage
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        qimage = QImage(buf, width, height, QImage.Format_ARGB32)
        # Convert QImage to QPixmap
        pixmap = QPixmap(qimage)
        return pixmap

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())









####################################
# Old code to find clusters of points

image_to_predict = tiff.imread(image_dir).transpose(2, 0, 1)
predicted_mask, uncertainty = seg.predict_image(image_to_predict, ['hand'], show=True)
predicted_mask[:,150:250,:] = 0

X,Y = np.where(predicted_mask[:,:,0]==255)

Z = np.column_stack((X,Y)).astype(np.float32)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,labels,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
color = labels.copy().astype(str).reshape(-1)
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