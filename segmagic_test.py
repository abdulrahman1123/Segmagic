from segmagic_ml_small import Segmagic
from tifffile import imread
import numpy as np
from skimage.measure import label, regionprops
import glob
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import os
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QButtonGroup,
                             QRadioButton,QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
                             QGridLayout,QDesktopWidget,QFrame,QSizePolicy)
from PyQt5.QtGui import QPixmap,QFont,QImage,QIcon
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

def find_intensity(image_dir,side_info, model_type):
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
    labels = ['MCP','IP','C'] if model_type == 'hand_p3' else [model_type]
    n_mask_regions = 18 if model_type== 'hand_p3' else 2

    # predict mask
    image_to_predict = imread(image_dir).transpose(2, 0, 1)
    predicted_mask, uncertainty = seg.predict_image(image_to_predict, labels, show=False)
    labeled_mask = label(predicted_mask)
    region_inds, region_count = np.unique(labeled_mask, return_counts=True)
    n_major_masks = len(np.where(region_count>1500)[0])-1
    
    # if the resulting mask is just one big mask, add a vertical line to split the image into two
    if n_major_masks==1:
        print("\nOne big region was found. Splitting using a vertical line\n")
        image_to_predict[:,:,190:210] = 255
        predicted_mask, uncertainty = seg.predict_image(image_to_predict, labels, show=False)
        labeled_mask, n_masks = label(predicted_mask, return_num=True)
    # Extract largest regions
    region_inds, region_count = np.unique(labeled_mask, return_counts=True)
    max_regions = np.min((len(region_count)-1,n_mask_regions))
    largest_regions = np.argsort(region_count)[::-1][1:max_regions+1]

    # choose the biggest 2 regions and label them
    filtered_mask = np.where(~np.isin(labeled_mask,largest_regions),0,labeled_mask)
    # sometimes there are small additional clusters that intefere with the numbering. E.g., you
    # might find that the largest clusters are 0,2 and 6. You want to convert that to 0,1 and 2
    for i in range(1,max_regions+1):
        filtered_mask [filtered_mask== np.unique(filtered_mask)[i]] = i

    filtered_regions = label(filtered_mask)
    
    img = image_to_predict.transpose((1,2,0))
    props_list = [regionprops(filtered_regions[:,:,i],img) for i in range(filtered_regions.shape[2])]

    # Extract centroids and side of affected extremity
    region_afctd_extr = []
    intensity_dic = {'id':image_dir.replace(".tif","").split("/")[-1].split("\\")[-1],
                     'ipsi':[],'contra':[],'ratio':[]}

    simple_filtered_mask = filtered_mask.copy()
    for nlayer,props in enumerate(props_list):
        centroids = [region.centroid[1] for region in props]
        left_centroids = np.argsort(centroids)[0:int(len(centroids)/2)]
        region_side = np.where(np.in1d(np.arange(len(centroids)),left_centroids),'left','right')
        #region_side = ["right" if centroid[1]>200 else "left" for centroid in centroids]
        region_afctd_extr = np.array(["ipsi" if side==afctd_side else "contra" for side in region_side])

        # make sure that those on the ipsilateral side are marked as 1 and the contrlateral as 2
        loc_filtered_mask = filtered_mask[:,:,nlayer].copy()
        ipsi_points = np.isin(loc_filtered_mask,np.unique(loc_filtered_mask)[1::][region_afctd_extr == 'ipsi'])
        contra_points = np.isin(loc_filtered_mask,np.unique(loc_filtered_mask)[1::][region_afctd_extr == 'contra'])
        simple_filtered_mask[:,:,nlayer][ipsi_points] = 1
        simple_filtered_mask[:,:,nlayer][contra_points] = 2
        # compute intensities
        intensity_dic['ipsi'].append(np.mean([255-np.mean(prop.intensity_mean) for prop in np.array(props)[region_afctd_extr == 'ipsi']]))
        intensity_dic['contra'].append(np.mean([255 - np.mean(prop.intensity_mean) for prop in np.array(props)[region_afctd_extr == 'contra']]))
        intensity_dic['ratio'].append(intensity_dic['ipsi'][-1] / intensity_dic['contra'][-1])

    intensity_dic['ipsi'] = ", ".join(np.round(np.array(intensity_dic['ipsi']),1).astype(str))
    intensity_dic['contra'] = ", ".join(np.round(np.array(intensity_dic['contra']),1).astype(str))
    intensity_dic['ratio'] = ", ".join(np.round(np.array(intensity_dic['ratio']),2).astype(str))

    filtered_mask = simple_filtered_mask[:,:,0]
    for layer in range(1,simple_filtered_mask.shape[2]):
        filtered_mask[simple_filtered_mask[:, :, layer] == 1] = 1
        filtered_mask[simple_filtered_mask[:, :, layer] == 2] = 2

    return image_to_predict,filtered_mask, centroids, region_afctd_extr, intensity_dic

# Choose by image
base_path = os.getcwd()
#data = pd.read_excel(base_path+'/pt_info.xlsx')
#image_dir = r"/home/abdulrahman/Downloads/CRPS Images/CRPS004_P3.tif"
#model_type = 'hand_p3'
#side_info = 'left'
#find_intensity(image_dir,side_info, model_type)

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
        # change window size depending on screen size
        sc_width = QDesktopWidget().screenGeometry(-1).width()
        self.MF  = 0.75*sc_width/1920 # magnification factor
        print(self.MF)
        self.MFF =0.6*sc_width/1920 # magnification factor for fonts

        # Create main layout
        self.main_layout = QHBoxLayout()

        self.left_layout = QVBoxLayout()
        self.output_layout = QVBoxLayout()
        self.dataselection_lo = QHBoxLayout()
        self.image_info_lo = QHBoxLayout()
        self.title_lo = QHBoxLayout()
        self.imageselection_lo = QHBoxLayout()
        self.folder_tootlip_lo = QHBoxLayout()
        self.folderselection_lo = QGridLayout()
        self.finalset_lo = QHBoxLayout()
        self.aff_side_lo = QGridLayout()
        self.handfoot_lo = QGridLayout()
        self.sciphase_lo = QGridLayout()
        #self.handfoot_lo.setContentsMargins(30, 0, 0, 0)
        
        self.title_label = QLabel("ScintiSeg", self)
        self.title_label.setWordWrap(True)
        self.title_label.setFont(QFont("Calibri", int(18*self.MFF)))
        self.title_label.setAlignment(Qt.AlignCenter)

        renderer = QPixmap(base_path+"/logo.png")
        logo_label = QLabel()
        logo_label.setFixedSize(int(75 * self.MF), int(75 * self.MF))
        logo_label.setPixmap(renderer.scaled(logo_label.size(), QtCore.Qt.KeepAspectRatio))

        single_img_label = QLabel("Segmenting a Single Image")
        single_img_label.setFont(QFont("Calibri", int(16 * self.MFF)))

        folder_seg_label = QLabel("Segmenting Multiple Images")
        folder_seg_label.setFont(QFont("Calibri", int(16 * self.MFF)))

        folder_seg_label_add = QLabel("(Hover over the question mark for more info)")
        folder_seg_label_add.setFont(QFont("Calibri", int(13 * self.MFF)))
        folder_seg_label_add.setWordWrap(True)

        tooltip_icon = QPixmap(base_path+"/question.png")
        tooltip_label = QLabel()
        tooltip_label.setFixedSize(int(80 * self.MF), int(40 * self.MF))
        tooltip_label.setPixmap(tooltip_icon.scaled(tooltip_label.size(), QtCore.Qt.KeepAspectRatio))
        tip = ("Choose a data set (excel file)\nIt should have three columns (ID, Side and Extremity), containing\ninformation about the images to be segmented.\nThe image naming should follow the following scheme:\n<subID_scintiPhase>.tif (e.g. CRPS006_P3.tif)")
        tooltip_label.setToolTip(f'<img src="{base_path+"/tooltip.png"}">')
        #tooltip_label.setToolTipDuration(5000)

        # TODO: make the label droppable
        label_txt = "For single-image analysis:\nEither drag and drop an image here,\nor use the 'Choose Image' button, and\npress 'Segment Image'"
        self.label1 = image_label(label_txt,self)
        #self.label1.setAcceptDrops(True)
        self.label1.setFixedSize(int(450*self.MF), int(450*self.MF))
        self.label1.setStyleSheet("background-color: white;border-style: solid; border-width: 1px; border-color: black")


        # Styles
        style_allround = "border-radius: 5px; background-color: lightgrey"
        style_right_round = "border-bottom-right-radius: 10px; border-top-right-radius: 10px; border-style: solid; border-width: 1px; border-color: black"
        style_left_round = "border-bottom-left-radius: 10px; border-top-left-radius: 10px; background-color: lightgrey;"
        
        # Create data selection widgets
        self.data_path_line = QLineEdit()
        self.data_path_line.setFixedHeight(int(44*self.MF))
        self.data_path_line.setStyleSheet(style_right_round)
        self.databrowse_button = QPushButton("Choose Data")
        self.databrowse_button.setStyleSheet(style_left_round)
        self.databrowse_button.setFixedSize(int(130*self.MF), int(44*self.MF))
        
        # For single-image mode, create radiobuttons and labels
        font_sub = QFont("Calibri", int(11*self.MFF))

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
        self.img_path_line = QLineEdit()
        self.img_path_line.setFixedHeight(int(44*self.MF))
        self.img_path_line.setStyleSheet(style_right_round)
        
        self.folder_path_line = QLineEdit()
        self.folder_path_line.setFixedHeight(int(44*self.MF))
        self.folder_path_line.setStyleSheet(style_right_round)

        
        self.folderbrowse_button = QPushButton("Choose Folder")
        self.folderbrowse_button.setStyleSheet(style_left_round)
        self.folderbrowse_button.setFixedSize(int(130*self.MF), int(44*self.MF))
        
        
        # Final set of buttons
        single_image_dir = base_path+"/single_image_button.png"
        multiple_images_dir = base_path+"/multiple_images_button.png"
        segment_image_dir = base_path+"/segment_image.png"
        segment_folder_dir = base_path+"/segment_folder.png"
        exit_dir = base_path+"/exit.png"

        self.segment_img_button = QPushButton()
        self.segment_img_button.setStyleSheet(style_allround)
        self.segment_img_button.setIcon(QIcon(segment_image_dir))
        self.segment_img_button.setIconSize(QtCore.QSize(int(100*self.MF),int(40*self.MF)))
        self.segment_img_button.setFixedHeight(int(44*self.MF))

        

        self.single_button = QPushButton()
        self.single_button.setStyleSheet(style_allround)
        self.single_button.setFixedHeight(int(44*self.MF))
        self.single_button.setIcon(QIcon(single_image_dir))
        self.single_button.setIconSize(QtCore.QSize(int(100*self.MF),int(40*self.MF)))

        self.multiple_button = QPushButton()
        self.multiple_button.setStyleSheet(style_allround)
        self.multiple_button.setFixedHeight(int(44*self.MF))
        self.multiple_button.setIcon(QIcon(multiple_images_dir))
        self.multiple_button.setIconSize(QtCore.QSize(int(100*self.MF),int(40*self.MF)))


        self.exit_button = QPushButton("")
        self.exit_button.setStyleSheet(style_allround)
        self.exit_button.setIcon(QIcon(exit_dir))
        self.exit_button.setIconSize(QtCore.QSize(int(65*self.MF),int(26*self.MF)))
        self.exit_button.setFixedHeight(int(44*self.MF))
        

        # Now the single-image-exclusive layout
        self.imagebrowse_button = QPushButton("Choose Image")
        self.imagebrowse_button.setStyleSheet(style_left_round)
        self.imagebrowse_button.setFixedSize(int(130*self.MF), int(44*self.MF))
        self.segment_folder_button = QPushButton("")
        self.segment_folder_button.setStyleSheet(style_allround)
        self.segment_folder_button.setIcon(QIcon(segment_folder_dir))
        self.segment_folder_button.setIconSize(QtCore.QSize(int(100*self.MF),int(40*self.MF)))
        self.segment_folder_button.setFixedHeight(int(88*self.MF))


        # Connect button click event to function
        self.folderbrowse_button.clicked.connect(self.browse_folder)
        self.imagebrowse_button.clicked.connect(self.browse_image)
        self.databrowse_button.clicked.connect(self.browse_data)
        self.segment_img_button.clicked.connect(self.segment_image)
        self.segment_folder_button.clicked.connect(self.segment_folder)
        self.exit_button.clicked.connect(self.close)
        
        self.createTable()
        
        # Build the layouts
        self.aff_side_lo.addWidget(self.aff_side_label,0,0,1,2)
        self.aff_side_lo.addWidget(self.aff_side_left,1,0,1,1)
        self.aff_side_lo.addWidget(self.aff_side_right,1,1,1,1)

        self.handfoot_lo.addWidget(self.handfoot_label,0,0,1,2)
        self.handfoot_lo.addWidget(self.handfoot_hand,1,0,1,1)
        self.handfoot_lo.addWidget(self.handfoot_foot,1,1,1,1)

        self.sciphase_lo.addWidget(self.sciphase_label,0,0,1,2)
        self.sciphase_lo.addWidget(self.sciphase_12,1,0,1,1)
        self.sciphase_lo.addWidget(self.sciphase_3,1,1,1,1)

        

        # Prepare the vertical separator
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.VLine)
        self.separator.setStyleSheet('color: lightgrey; background-color: transparent')
        self.separator.setLineWidth(int(10*self.MF))

        # prepare the horizontal separator
        Separador1 = QFrame()
        Separador1.setFrameShape(QFrame.HLine)  # Set the shape to horizontal line
        Separador1.setStyleSheet('color: lightgrey')
        Separador1.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        Separador1.setLineWidth(int(1 * self.MF))

        Separador2 = QFrame()
        Separador2.setFrameShape(QFrame.HLine)  # Set the shape to horizontal line
        Separador2.setStyleSheet('color: lightgrey')
        Separador2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        Separador2.setLineWidth(int(1 * self.MF))

        # Add all widgets to thier respective layouts
        self.output_layout.addWidget(self.label1)
        self.output_layout.addWidget(self.tableWidget)

        self.image_info_lo.addLayout(self.aff_side_lo)
        self.image_info_lo.addWidget(self.spacer_label)
        self.image_info_lo.addLayout(self.handfoot_lo)
        self.image_info_lo.addWidget(self.spacer_label)
        self.image_info_lo.addLayout(self.sciphase_lo)
        

        self.imageselection_lo.addWidget(self.imagebrowse_button)
        self.imageselection_lo.addWidget(self.img_path_line)
        self.imageselection_lo.addWidget(self.spacer_label)
        self.imageselection_lo.addWidget(self.segment_img_button)
        self.imageselection_lo.setSpacing(0)

        self.folderselection_lo.addWidget(self.segment_folder_button,0,5,2,1)
        self.folderselection_lo.addWidget(self.folderbrowse_button,0,0,1,1)
        self.folderselection_lo.addWidget(self.folder_path_line,0,1,1,3)
        self.folderselection_lo.addWidget(self.spacer_label,0,4,1,1)
        self.folderselection_lo.addWidget(self.databrowse_button,1,0,1,1)
        self.folderselection_lo.addWidget(self.data_path_line,1,1,1,3)
        self.folderselection_lo.setHorizontalSpacing(0)

        self.finalset_lo.addWidget(self.exit_button)
        self.finalset_lo.addWidget(self.spacer_label)
        self.finalset_lo.addWidget(self.spacer_label)
        self.finalset_lo.addWidget(self.spacer_label)
        self.finalset_lo.setContentsMargins(0, 50, 0, 0)
        
        self.title_lo.addWidget(logo_label)
        self.title_lo.addWidget(self.title_label)

        self.folder_tootlip_lo.addWidget(folder_seg_label)
        self.folder_tootlip_lo.addWidget(tooltip_label)
        #self.folder_tootlip_lo.addWidget(self.separator)

        # Add all Layouts to the main one
        self.left_layout.addLayout(self.title_lo)
        self.left_layout.addWidget(Separador1)
        self.left_layout.addWidget(single_img_label)
        self.left_layout.addLayout(self.image_info_lo)
        self.left_layout.addLayout(self.imageselection_lo)
        self.left_layout.addWidget(Separador2)
        #self.left_layout.addWidget(folder_seg_label)
        self.left_layout.addLayout(self.folder_tootlip_lo)
        self.left_layout.addLayout(self.dataselection_lo)
        self.left_layout.addLayout(self.folderselection_lo)
        self.left_layout.addLayout(self.finalset_lo)
        
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.output_layout)
        
        # Set the layout
        self.setLayout(self.main_layout)        
        for i in range(self.main_layout.count()):
            item = self.main_layout.itemAt(i)
            #print(item)

    def build_multiple_lo(self):
        # OLD FUNCTION ... IGNORE
        clear_layout(self.main_layout)
        #self.title_label.setText("ScintiSig")
        self.createTable()

        # Add all widgets to thier respective layouts
        self.output_layout.addWidget(self.tableWidget)
        self.dataselection_lo.addWidget(self.data_path_line)
        self.dataselection_lo.addWidget(self.databrowse_button)
        self.imageselection_lo.addWidget(self.img_path_line)
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
        ipsi = QTableWidgetItem(content['ipsi'])
        ipsi.setTextAlignment(Qt.AlignCenter)
        contra = QTableWidgetItem(content['contra'])
        contra.setTextAlignment(Qt.AlignCenter)
        ratio = QTableWidgetItem(content['ratio'])
        ratio.setTextAlignment(Qt.AlignCenter)

        id = QTableWidgetItem(content['id'])
        id.setTextAlignment(Qt.AlignCenter)
        self.tableWidget.setItem(current_row_count, 0, id)
        self.tableWidget.setItem(current_row_count, 1, ipsi)
        self.tableWidget.setItem(current_row_count, 2, contra)
        self.tableWidget.setItem(current_row_count, 3, ratio)
  
        
    def createTable(self):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(4) 
        self.tableWidget.setHorizontalHeaderLabels(["ID","Ipsi", "Contra","Ratio"])
        self.tableWidget.setMinimumHeight(int(200*self.MF))
        #self.tableWidget.horizontalHeader().setStretchLastSection(True)  # Stretch the last section
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Resize mode
        
        #self.tableWidget.resizeColumnsToContents()

        

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
            self.img_path_line.setText(file_path)
            self.load_image()

    def load_image(self):
        image_path = self.img_path_line.text()
        pixmap = QPixmap(image_path)
        # Set the pixmap to the first QLabel
        self.label1.setPixmap(pixmap.scaled(self.label1.size(), QtCore.Qt.KeepAspectRatio))

    def segment_folder(self):
        data = pd.read_excel(self.data_path_line.text())
        files_dir = glob.glob(self.folder_path_line.text()+"/*")
        print(files_dir)
        if len(files_dir)>0:
            for image_dir in files_dir:
                if os.path.exists(image_dir):
                    id,phase = image_dir.replace(".tif","").split("/")[-1].split("\\")[-1].split("_")
                    extremity = data.loc[data['ID']==id,'Extremity'].values[0].lower()
                    model_type = 'hand' if 'up' in extremity else 'foot'

                    # hand_p3 is a special model for the hands
                    model_type = 'hand_p3'if model_type+phase == 'hand3' else model_type
                    side_info = data.loc[data['ID']==id,'Side'].values[0].lower()
                    image_to_predict,filtered_mask, centroids, region_afctd_extr, intensity_dic = find_intensity(image_dir,side_info, model_type)
                    self.editTable(intensity_dic)

            # plot the last image only
            color_list = ['steelblue', 'indianred', 'olivedrab', 'darkgoldenrod', 'darkmagenta', 'grey', 'palevioletred',
                          'sienna', 'beige', 'coral']

            cmap = plt.cm.colors.ListedColormap(['white'] + color_list[0:len(centroids)])
            plt.close()
            fig = plt.subplots(figsize=(5, 5))
            plt.imshow(image_to_predict[0, :, :], cmap='gray')
            plt.imshow(filtered_mask, cmap=cmap, interpolation='nearest', alpha=0.3)
            # Add text
            for centroid, l_text in zip(centroids, region_afctd_extr):
                plt.text(centroid[1], centroid[0], l_text, ha='center', font='Calibri', size=20)

            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            plt.savefig(base_path + "/temp_file.png")
            plt.close()
            pixmap_regions = QPixmap(base_path + "/temp_file.png")
            self.label1.setPixmap(pixmap_regions.scaled(self.label1.size(), QtCore.Qt.KeepAspectRatio))
    def segment_image(self):
        image_dir = self.img_path_line.text()
        if os.path.exists(image_dir):
            side_info = "left" if self.aff_side_left.isChecked() else "right"
            model_type = "hand" if self.handfoot_hand.isChecked() else "foot"
            phase = "3" if self.sciphase_3.isChecked() else "12"
            id = image_dir.replace(".tif","").split("/")[-1].split("\\")[-1]
            
            model_type = 'hand_p3'if model_type+phase == 'hand3' else model_type
            image_to_predict,filtered_mask, centroids, region_afctd_extr, intensity_dic = find_intensity(image_dir,side_info, model_type)
            color_list = ['steelblue', 'indianred','olivedrab','darkgoldenrod','darkmagenta','grey','palevioletred','sienna','beige','coral']

            cmap = plt.cm.colors.ListedColormap(['white']+color_list[0:len(centroids)])
            plt.close()
            fig = plt.subplots(figsize = (5,5))
            plt.imshow(image_to_predict[0,:,:],cmap='gray')
            plt.imshow(filtered_mask, cmap=cmap, interpolation='nearest', alpha = 0.3)
            # Add text
            #for centroid,l_text in zip(centroids,region_afctd_extr):
            #    plt.text(centroid[1],centroid[0],l_text, ha='center', font = 'Calibri', size = 20)

            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            plt.savefig(base_path+"/temp_file.png")
            plt.close()
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
class image_label(QLabel):
    def __init__(self, title, parent):
        super().__init__(title, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            img_path = m.urls()[0].toLocalFile()
            pixmap = QPixmap(img_path)
            self.setPixmap(pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio))
            self.parent().img_path_line.setText(m.urls()[0].toLocalFile())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

