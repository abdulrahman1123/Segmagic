from segmagic_ml import Segmagic
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

import pandas as pd
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QButtonGroup, QRadioButton,QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout,QLayout,QFrame
from PyQt5.QtGui import QPixmap,QFont,QImage,QIcon
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import os
import glob

def find_intensity(image_dir,side_info, id, model_type):
    """
    Find segments of ipsilateral and contralateral sides and calculate the intensity inside each region.
    :param image_dir: directory of the image to be segmented
    :param side_info: a string representing the effected side (can only be "right" or "left").
    :return: a tuple of 5 objects: image_to_predict as numpy array, the filtered_mask, a list of centroids,
             the label for each segment produced ("ipsi" and "contra") corresponding to the segments; and a 
             dictionary that has intensity information
    """
    print(base_path+f'/{model_type}')
    seg = Segmagic(base_path+f'/{model_type}')
    afctd_side = side_info.lower()
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

#find_intensity(image_dir,data,id, model_type)

def clear_layout(layout):
    for i in reversed(range(layout.count())):
        loc_widget = layout.itemAt(i).widget()
        loc_layout = layout.itemAt(i).layout()

        if loc_widget is not None or isinstance(layout.itemAt(i),QFrame):
            loc_widget.setParent(None)
        elif loc_layout is not None:
            clear_layout(loc_layout)
            #layout.removeItem(item)


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
        self.title_lo = QHBoxLayout()
        self.imageselection_lo = QHBoxLayout()
        self.finalset_lo = QHBoxLayout()
        self.aff_side_lo = QGridLayout()
        self.handfoot_lo = QGridLayout()
        self.sciphase_lo = QGridLayout()
        #self.handfoot_lo.setContentsMargins(30, 0, 0, 0)
        
        self.title_label = QLabel("ScintiSeg", self)
        self.title_label.setWordWrap(True)
        self.title_label.setFont(QFont("Calibri", 20))
        self.title_label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap(base_path+"/logo.png")
        self.logo_label = QLabel()
        self.logo_label.setFixedSize(75, 75)
        self.logo_label.setPixmap(pixmap.scaled(self.logo_label.size(), QtCore.Qt.KeepAspectRatio))

        # TODO: make the label droppable
        self.label1 = QLabel("Drag and drop an image here, or use the button below")
        #self.label1.setAcceptDrops(True)
        self.label1.setFixedSize(350, 350)
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
        font_sub = QFont("Calibri", 12)

        data_font = QFont("Calibri", 12)
        self.aff_side_label = QLabel("Affected side")
        self.aff_side_left = QRadioButton("Left")
        self.aff_side_left.setChecked(True)
        self.aff_side_right = QRadioButton("Right")
        self.aff_side_label.setFont(font_sub)
        self.aff_side_left.setFont(font_sub)
        self.aff_side_right.setFont(font_sub)
        self.aff_side_but_group = QButtonGroup(self)
        self.aff_side_but_group.addButton(self.aff_side_left)
        self.aff_side_but_group.addButton(self.aff_side_right)


        self.handfoot_label = QLabel("Affected part")
        self.handfoot_hand = QRadioButton("Hand")
        self.handfoot_hand.setChecked(True)
        self.handfoot_foot = QRadioButton("Foot")
        #self.handfoot_label.setAlignment(Qt.AlignCenter)
        self.handfoot_label.setFont(font_sub)
        self.handfoot_hand.setFont(font_sub)
        self.handfoot_foot.setFont(font_sub)
        self.handfoot_but_group = QButtonGroup(self)
        self.handfoot_but_group.addButton(self.handfoot_hand)
        self.handfoot_but_group.addButton(self.handfoot_foot)


        self.sciphase_label = QLabel("Scinti. phase")
        self.sciphase_12 = QRadioButton("1 or 2")
        self.sciphase_12.setChecked(True)
        self.sciphase_3 = QRadioButton("3")
        #self.sciphase_label.setAlignment(Qt.AlignCenter)
        self.sciphase_label.setFont(font_sub)
        self.sciphase_12.setFont(font_sub)
        self.sciphase_3.setFont(font_sub)
        self.sciphase_but_group = QButtonGroup(self)
        self.sciphase_but_group.addButton(self.sciphase_12)
        self.sciphase_but_group.addButton(self.sciphase_3)
        
        self.spacer_label = QLabel("   ")

        # Create image selection widgets
        self.path_line = QLineEdit()
        self.path_line.setFixedHeight(30)
        self.path_line.setStyleSheet(style_left_round)
        self.folderbrowse_button = QPushButton("Open Folder")
        self.folderbrowse_button.setStyleSheet(style_right_round)
        self.folderbrowse_button.setFixedSize(100, 30)
        
        
        # Final set of buttons
        single_image_dir = base_path+"/single_image_button.png"
        multiple_images_dir = base_path+"/multiple_images_button.png"
        segment_image_dir = base_path+"/segment_image.png"
        segment_folder_dir = base_path+"/segment_folder.png"
        exit_dir = base_path+"/exit.png"

        self.segment_img_button = QPushButton()
        self.segment_img_button.setStyleSheet(style_allround)
        self.segment_img_button.setIcon(QIcon(segment_image_dir))
        self.segment_img_button.setIconSize(QtCore.QSize(100,40))
        self.segment_img_button.setFixedHeight(44)

        

        self.single_button = QPushButton()
        self.single_button.setStyleSheet(style_allround)
        self.single_button.setFixedHeight(44)
        self.single_button.setIcon(QIcon(single_image_dir))
        self.single_button.setIconSize(QtCore.QSize(100,40))

        self.multiple_button = QPushButton()
        self.multiple_button.setStyleSheet(style_allround)
        self.multiple_button.setFixedHeight(44)
        self.multiple_button.setIcon(QIcon(multiple_images_dir))
        self.multiple_button.setIconSize(QtCore.QSize(100,40))


        self.exit_button = QPushButton("")
        self.exit_button.setStyleSheet(style_allround)
        self.exit_button.setIcon(QIcon(exit_dir))
        self.exit_button.setIconSize(QtCore.QSize(65,26))
        self.exit_button.setFixedHeight(44)
        

        # Now the single-image-exclusive layout
        self.imagebrowse_button = QPushButton("Open Image")
        self.imagebrowse_button.setStyleSheet(style_right_round)
        self.imagebrowse_button.setFixedSize(100, 30)
        self.segment_folder_button = QPushButton("")
        self.segment_folder_button.setStyleSheet(style_allround)
        self.segment_folder_button.setIcon(QIcon(segment_folder_dir))
        self.segment_folder_button.setIconSize(QtCore.QSize(100,40))
        self.segment_folder_button.setFixedHeight(44)


        # Connect button click event to function
        self.folderbrowse_button.clicked.connect(self.browse_folder)
        self.imagebrowse_button.clicked.connect(self.browse_image)
        self.databrowse_button.clicked.connect(self.browse_data)
        self.segment_img_button.clicked.connect(self.segment_image)
        self.segment_folder_button.clicked.connect(self.segment_folder)
        self.exit_button.clicked.connect(self.close)
        self.single_button.clicked.connect(self.build_single_lo)
        self.multiple_button.clicked.connect(self.build_multiple_lo)
        
        self.mode = 'single'
        self.createTable(mode = self.mode)
        
        # Build the layouts
        
        if self.mode == 'single':
            self.build_single_lo()
        elif self.mode == 'multiple':
            self.build_multiple_lo()
        else:
            print("The layout mode should be either 'single' or 'multiple'")

        """
        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)"""
    
    def build_single_lo(self):
        clear_layout(self.main_layout)
        #self.clearLayout(self.main_layout)
        self.aff_side_lo.addWidget(self.aff_side_label,0,0,1,2)
        self.aff_side_lo.addWidget(self.aff_side_left,1,0,1,1)
        self.aff_side_lo.addWidget(self.aff_side_right,1,1,1,1)

        self.handfoot_lo.addWidget(self.handfoot_label,0,0,1,2)
        self.handfoot_lo.addWidget(self.handfoot_hand,1,0,1,1)
        self.handfoot_lo.addWidget(self.handfoot_foot,1,1,1,1)

        self.sciphase_lo.addWidget(self.sciphase_label,0,0,1,2)
        self.sciphase_lo.addWidget(self.sciphase_12,1,0,1,1)
        self.sciphase_lo.addWidget(self.sciphase_3,1,1,1,1)
        



        self.mode = 'single'
        self.createTable(mode = self.mode)

        # Prepare the separator
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.VLine)
        self.separator.setStyleSheet('color: lightgrey; background-color: transparent')
        self.separator.setLineWidth(10)

        # Add all widgets to thier respective layouts
        self.output_layout.addWidget(self.label1)
        self.output_layout.addWidget(self.tableWidget)
        self.dataselection_lo.addLayout(self.aff_side_lo)
        self.dataselection_lo.addWidget(self.spacer_label)
        self.dataselection_lo.addLayout(self.handfoot_lo)
        self.dataselection_lo.addWidget(self.spacer_label)
        self.dataselection_lo.addLayout(self.sciphase_lo)
        self.imageselection_lo.addWidget(self.path_line)
        self.imageselection_lo.addWidget(self.imagebrowse_button)
        self.finalset_lo.addWidget(self.segment_img_button)
        self.finalset_lo.addWidget(self.multiple_button)
        self.finalset_lo.addWidget(self.exit_button)
        self.title_lo.addWidget(self.logo_label)
        self.title_lo.addWidget(self.title_label)

        # Add all Layouts to the main one
        self.main_layout.addLayout(self.title_lo)
        self.main_layout.addLayout(self.output_layout)
        self.main_layout.addLayout(self.dataselection_lo)
        self.main_layout.addLayout(self.imageselection_lo)
        self.main_layout.addLayout(self.finalset_lo)

        # Set the layout
        self.setLayout(self.main_layout)        
        for i in range(self.main_layout.count()):
            item = self.main_layout.itemAt(i)
            #print(item)
            
    def build_multiple_lo(self):
        clear_layout(self.main_layout)
        #self.title_label.setText("ScintiSig")
        self.mode = 'multiple'
        self.createTable(mode = self.mode)

        # Add all widgets to thier respective layouts
        self.output_layout.addWidget(self.tableWidget)
        self.dataselection_lo.addWidget(self.data_path_line)
        self.dataselection_lo.addWidget(self.databrowse_button)
        self.imageselection_lo.addWidget(self.path_line)
        self.imageselection_lo.addWidget(self.folderbrowse_button)
        self.finalset_lo.addWidget(self.segment_folder_button)
        self.finalset_lo.addWidget(self.single_button)
        self.finalset_lo.addWidget(self.exit_button)

        
        self.title_lo.addWidget(self.logo_label)
        self.title_lo.addWidget(self.title_label)

        # Add all Layouts to the main one
        self.main_layout.addLayout(self.title_lo)
        self.main_layout.addLayout(self.output_layout)
        self.main_layout.addLayout(self.dataselection_lo)
        self.main_layout.addLayout(self.imageselection_lo)
        self.main_layout.addLayout(self.finalset_lo)

        # Set the layout
        self.setLayout(self.main_layout)

        
    def editTable(self, content):
        current_row_count = self.tableWidget.rowCount()
        self.tableWidget.insertRow(current_row_count)
        ipsi = QTableWidgetItem(str(np.round(content['ipsi'],1)))
        ipsi.setTextAlignment(Qt.AlignCenter)
        contra = QTableWidgetItem(str(np.round(content['contra'],1)))
        contra.setTextAlignment(Qt.AlignCenter)
        ratio = QTableWidgetItem(str(np.round(content['ratio'],1)))
        ratio.setTextAlignment(Qt.AlignCenter)

        if self.mode=='multiple':
            id = QTableWidgetItem(str(np.round(content['id'],1)))
            id.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(current_row_count, 0, id)
            self.tableWidget.setItem(current_row_count, 1, ipsi)
            self.tableWidget.setItem(current_row_count, 2, contra)
            self.tableWidget.setItem(current_row_count, 3, ratio)
        else:
            self.tableWidget.setItem(current_row_count, 0, ipsi)
            self.tableWidget.setItem(current_row_count, 1, contra)
            self.tableWidget.setItem(current_row_count, 2, ratio)

        
    def createTable(self, mode='single'):
        self.tableWidget = QTableWidget()

        if mode == 'single':
            self.tableWidget.setRowCount(0)
            self.tableWidget.setColumnCount(3) 
            self.tableWidget.setHorizontalHeaderLabels(["Ipsi", "Contra","Ratio"])
            self.tableWidget.setFixedHeight(60)
        else:
            self.tableWidget.setRowCount(0)
            self.tableWidget.setColumnCount(4) 
            self.tableWidget.setHorizontalHeaderLabels(["ID","Ipsi", "Contra","Ratio"])
            self.tableWidget.setFixedHeight(450)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)  # Stretch the last section
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Resize mode

        

    def browse_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select data frame", "", "Images (*.xlsx);;All Files (*)", options=options)
        if file_path:
            self.data_path_line.setText(file_path)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Images Folder")
        if folder_path:
            self.path_line.setText(folder_path)

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
        files_dir = glob.glob(self.path_line.text()+"/*")
        print(files_dir)
        for image_dir in files_dir:
            if os.path.exists(image_dir):
                id,model_type,phase = image_dir.replace(".tif","").split("/")[-1].split("\\")[-1].split("_")
                
                model_type = 'hand_p3'if model_type+phase == 'hand3' else model_type
                side_info = data.loc[data['ID']==id,'affected_extremity'].values[0].lower()
                image_to_predict,filtered_mask, centroids, region_afctd_extr, intensity_dic = find_intensity(image_dir,side_info,id, model_type)
                self.editTable(intensity_dic)

    def segment_image(self):
        image_dir = self.path_line.text()
        if os.path.exists(image_dir):
            side_info = "left" if self.aff_side_left.isChecked() else "right"
            model_type = "hand" if self.handfoot_hand.isChecked() else "foot"
            phase = "3" if self.sciphase_3.isChecked() else "12"
            id = image_dir.replace(".tif","").split("/")[-1].split("\\")[-1]
            
            model_type = 'hand_p3'if model_type+phase == 'hand3' else model_type
            image_to_predict,filtered_mask, centroids, region_afctd_extr, intensity_dic = find_intensity(image_dir,side_info,id, model_type)
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

