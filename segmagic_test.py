from segmagic_ml import Segmagic
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

import pandas as pd
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton,QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout,QLayout
from PyQt5.QtGui import QPixmap,QFont,QImage,QIcon
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import os


def find_intensity(image_dir,data):
    name_split = image_dir.replace(".tif","").split("/")[-1].split("\\")[-1].split("_")
    if len(name_split)!=3:
        raise Warning(f"Make sure to use the following naming scheme 'subjectid_extremity_phase#' (e.g. CRPS001_hand_3).\n\
                    The given name is {'_'.join(name_split)}")
    # Extract importat information
    id, model_type, phase = name_split
    model_type = 'hand_p3'if model_type+phase == 'hand3' else model_type
    seg = Segmagic(base_path+f'/{model_type}')
    afctd_side = data.loc[data['ID']==id,'affected_extremity'].values[0].lower()
    labels = [model_type]

    # predict mask
    image_to_predict = imread(image_dir).transpose(2, 0, 1)
    predicted_mask, uncertainty = seg.predict_image(image_to_predict, labels, show=True)
    labeled_mask = label(predicted_mask)
    region_inds, region_count = np.unique(labeled_mask, return_counts=True)
    n_masks = len(np.where(region_count>1500)[0])-1
    
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
    filtered_mask [filtered_mask== np.unique(filtered_mask)[1]] = 1 # make sure you have only 0, 1 and 2
    filtered_mask [filtered_mask== np.unique(filtered_mask)[2]] = 2
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
base_path = os.getcwd()
#data = pd.read_excel(base_path+'/pt_info.xlsx')
#image_dir = r"\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\opg paper\Segmagic\all_images\CRPS007_foot_1.tif" #CRPS007P3#CRPS004P1

#find_intensity(image_dir,data)
def clear_layout(layout):
    for i in reversed(range(layout.count())):
        item = layout.itemAt(i)
        if isinstance(item, QWidget):
            widget = item.widget()
            widget.deleteLater()
        elif isinstance(item, QLayout):
            clear_layout(item)
            layout.removeItem(item)


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(r'"\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Downloads\_c19b537e-b81f-4d8f-bc65-893fa21adb77.jfif"'))
        
        self.mode = 'single'

        # Create main layout
        self.main_layout = QVBoxLayout()

        # Create QHBoxLayout for labels
        self.output_layout = QVBoxLayout()
        self.dataselection_lo = QHBoxLayout()
        self.imageselection_lo = QHBoxLayout()
        self.finalset_lo = QHBoxLayout()

        sub_text = "Choose the image to be segmented and intensity calculated.\nThe image file should have the following naming scheme 'subjectid_extremity_phase#' (e.g. CRPS001_hand_3)"
        self.info_label = QLabel(sub_text, self)
        self.info_label.setWordWrap(True)
        font_sub = QFont("Calibri", 12)
        self.info_label.setFont(font_sub)
        self.info_label.setAlignment(Qt.AlignCenter)  # Align text to the center

        # TODO: make the label droppable
        self.label1 = QLabel()
        #self.label1.setAcceptDrops(True)
        self.label1.setFixedSize(400, 400)
        self.label1.setStyleSheet("background-color: white;border-style: solid; border-width: 1px; border-color: black")


        # Styles
        style_allround = "border-radius: 5px; background-color: lightgrey"
        style_right_round = "border-bottom-right-radius: 10px; border-top-right-radius: 10px; background-color: lightgrey"
        style_left_round = "border-bottom-left-radius: 10px; border-top-left-radius: 10px; border-style: solid; border-width: 1px; border-color: black;"
        
        # Create data selection widgets
        self.data_path_line = QLineEdit()
        self.data_path_line.setFixedHeight(30)
        self.data_path_line.setStyleSheet(style_left_round)
        self.databrowse_button = QPushButton("Load data frame")
        self.databrowse_button.setStyleSheet(style_right_round)
        self.databrowse_button.setFixedSize(100, 30)
        
        # For single-image mode, create radiobuttons and labels
        self.aff_side_label = QLabel("Affected side")
        self.aff_side_left = QRadioButton("Left")
        self.aff_side_right = QRadioButton("Right")
        
        self.handfoot_label = QLabel("Affected part")
        self.handfoot_hand = QRadioButton("Hand")
        self.handfoot_foot = QRadioButton("Foot")
                
        # Create image selection widgets
        self.path_line = QLineEdit()
        self.path_line.setFixedHeight(30)
        self.path_line.setStyleSheet(style_left_round)
        self.folderbrowse_button = QPushButton("Open Folder")
        self.folderbrowse_button.setStyleSheet(style_right_round)
        self.folderbrowse_button.setFixedSize(100, 30)
        
        # Final set of buttons
        self.segment_img_button = QPushButton("Segment")
        self.exit_button = QPushButton("Exit")
        self.segment_img_button.setStyleSheet(style_allround)
        self.exit_button.setStyleSheet(style_allround)
        self.segment_img_button.setFixedHeight(30)
        self.exit_button.setFixedHeight(30)
        

        # Now the single-image-exclusive layout
        self.imagebrowse_button = QPushButton("Open Image")
        self.imagebrowse_button.setStyleSheet(style_right_round)
        self.imagebrowse_button.setFixedSize(100, 30)
        self.segment_folder_button = QPushButton("Segment Folder")
        self.segment_folder_button.setStyleSheet(style_allround)
        self.segment_folder_button.setFixedHeight(30)


        # Connect button click event to function
        self.folderbrowse_button.clicked.connect(self.browse_folder)
        self.imagebrowse_button.clicked.connect(self.browse_image)
        self.databrowse_button.clicked.connect(self.browse_data)
        self.segment_img_button.clicked.connect(self.segment_image)
        self.segment_folder_button.clicked.connect(self.segment_folder)
        self.exit_button.clicked.connect(self.close)
        
        self.createTable()
        
        # Build the layouts
        if self.mode=='multiple':
            self.dataselection_lo.addWidget(self.data_path_line)
            self.dataselection_lo.addWidget(self.databrowse_button)
            self.imageselection_lo.addWidget(self.path_line)
            self.imageselection_lo.addWidget(self.folderbrowse_button)
            self.finalset_lo.addWidget(self.segment_folder_button)
            self.finalset_lo.addWidget(self.exit_button)
        self.build_single_lo()
        """
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)"""

    def build_single_lo(self):
        clear_layout(self.main_layout)
        self.aff_side_lo = QGridLayout()
        self.aff_side_lo.addWidget(self.aff_side_label,0,0,2,1)
        self.aff_side_lo.addWidget(self.aff_side_left,1,0,1,1)
        self.aff_side_lo.addWidget(self.aff_side_right,1,1,1,1)

        self.handfoot_lo = QGridLayout()
        self.handfoot_lo.addWidget(self.handfoot_label,0,0,2,1)
        self.handfoot_lo.addWidget(self.handfoot_hand,1,0,1,1)
        self.handfoot_lo.addWidget(self.handfoot_foot,1,1,1,1)
        self.dataselection_lo.addLayout(self.databrowse_button)

        self.dataselection_lo.addLayout(self.databrowse_button)
        self.imageselection_lo.addWidget(self.path_line)
        self.imageselection_lo.addWidget(self.imagebrowse_button)
        self.finalset_lo.addWidget(self.segment_img_button)
        self.finalset_lo.addWidget(self.exit_button)
        self.output_layout.addWidget(self.label1)

        

        # Add QHBoxLayout, button, and QLineEdit to QVBoxLayout
        self.output_layout.addWidget(self.tableWidget)
        self.main_layout.addWidget(self.info_label)
        self.main_layout.addLayout(self.output_layout)
        self.main_layout.addLayout(self.dataselection_lo)
        self.main_layout.addLayout(self.imageselection_lo)
        self.main_layout.addLayout(self.finalset_lo)

        # Prepare the separator
        #separator = QFrame()
        #separator.setFrameShape(QFrame.VLine)
        #separator.setStyleSheet('color: lightgrey; background-color: transparent')
        #separator.setLineWidth(10)


        # Set the layout
        self.setLayout(self.main_layout)
        
        

    def build_multiple_lo(self):
        self.dataselection_lo.addWidget(self.databrowse_button)
        self.imageselection_lo.addWidget(self.path_line)
        self.imageselection_lo.addWidget(self.imagebrowse_button)
        self.finalset_lo.addWidget(self.segment_img_button)
        self.finalset_lo.addWidget(self.exit_button)
        self.output_layout.addWidget(self.label1)

        

        # Add QHBoxLayout, button, and QLineEdit to QVBoxLayout
        self.output_layout.addWidget(self.tableWidget)
        self.main_layout.addWidget(self.info_label)
        self.main_layout.addLayout(self.output_layout)
        self.main_layout.addLayout(self.dataselection_lo)
        self.main_layout.addLayout(self.imageselection_lo)
        self.main_layout.addLayout(self.finalset_lo)

        # Prepare the separator
        #separator = QFrame()
        #separator.setFrameShape(QFrame.VLine)
        #separator.setStyleSheet('color: lightgrey; background-color: transparent')
        #separator.setLineWidth(10)


        # Set the layout
        self.setLayout(self.main_layout)

        
    def editTable(self, content):
        current_row_count = self.tableWidget.rowCount()
        self.tableWidget.insertRow(current_row_count)
        self.tableWidget.setItem(0, 0, QTableWidgetItem(str(np.round(content['ipsi'],1))))  
        self.tableWidget.setItem(0, 1, QTableWidgetItem(str(np.round(content['contra'],1))))
        self.tableWidget.setItem(0, 2, QTableWidgetItem(str(np.round(content['ratio'],2))))
        
    def createTable(self):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(3) 
        self.tableWidget.setHorizontalHeaderLabels(["Ipsi", "Contra","Ratio"])
        
        self.tableWidget.horizontalHeader().setStretchLastSection(True)  # Stretch the last section
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Resize mode
        if self.mode == 'single':
            self.tableWidget.setFixedHeight(70)
        else:
            self.tableWidget.setFixedHeight(400)
        

    def browse_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select data frame", "", "Images (*.xlsx);;All Files (*)", options=options)
        if file_path:
            self.data_path_line.setText(file_path)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Images Folder")
        if folder_path:
            self.folder_path_line.setText(folder_path)

    def browse_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.tif);;All Files (*)", options=options)
        if file_path:
            self.path_line.setText(file_path)
            self.load_image()

    def load_image(self):
        image_path = self.path_line.text()
        pixmap = QPixmap(image_path)
        # Set the pixmap to the first QLabel
        self.label1.setPixmap(pixmap.scaled(self.label1.size(), QtCore.Qt.KeepAspectRatio))

    def segment_folder(self):
        data = pd.read_excel(self.data_path_line.text())
        files_dir = self.path_line.text()
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
            self.label1.setPixmap(pixmap_regions.scaled(self.label1.size(), QtCore.Qt.KeepAspectRatio))
            self.editTable(intensity_dic)

    def segment_image(self):
        data = pd.read_excel(self.data_path_line.text())
        image_dir = self.path_line.text()
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
            self.label1.setPixmap(pixmap_regions.scaled(self.label1.size(), QtCore.Qt.KeepAspectRatio))
            self.editTable(intensity_dic)

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

