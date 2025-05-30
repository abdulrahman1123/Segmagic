from segmagic_ml_small import Segmagic
from PIL import Image
import numpy as np
from skimage.measure import label, regionprops
import glob
import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


import os
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QButtonGroup,
                             QRadioButton,QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
                             QGridLayout,QFrame,QDesktopWidget,QSizePolicy, QAbstractScrollArea)
from qtwidgets import AnimatedToggle
from PyQt5.QtGui import QPixmap,QFont,QImage,QIcon,QGuiApplication
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from matplotlib import rcsetup

from inspect import getsourcefile



def find_intensity(image_dir,side_info, model_type, fast_mode):
    """
    Find segments of ipsilateral and contralateral sides and calculate the intensity inside each region.
    :param image_dir: directory of the image to be segmented
    :param side_info: a string representing the effected side (can only be "right" or "left").
    :return: a tuple of 5 objects: image_to_predict as numpy array, the filtered_mask, a list of centroids,
             the label for each segment produced ("ipsi" and "contra") corresponding to the segments; and a 
             dictionary that has intensity information
    """
    seg = Segmagic(base_path+f'/{model_type}',fast_mode)
    afctd_side = side_info.lower()
    labels = ['MCP','IP','C'] if model_type == 'hand_p3' else [model_type]
    n_mask_regions = 18 if model_type== 'hand_p3' else 2

    # predict mask
    #image_to_predict = imread(image_dir).transpose(2, 0, 1)
    image_to_predict = np.array(Image.open(image_dir)).transpose(2, 0, 1)
    image_to_predict = image_to_predict[0:3,:,:] # some images have a 4th layer representing transparency (e.g., png images)
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
    intensity_dic = {'id':".".join(image_dir.split(".")[0:-1]).split("/")[-1].split("\\")[-1],
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
    intensity_dic['ratio'] = ", ".join(np.round(np.array(intensity_dic['ratio']),1).astype(str))

    filtered_mask = simple_filtered_mask[:,:,0]
    for layer in range(1,simple_filtered_mask.shape[2]):
        filtered_mask[simple_filtered_mask[:, :, layer] == 1] = 1
        filtered_mask[simple_filtered_mask[:, :, layer] == 2] = 2
    ipsi_cent = (np.median(np.where(filtered_mask==1)[0]),np.median(np.where(filtered_mask==1)[1]))
    contr_cent = (np.median(np.where(filtered_mask==2)[0]),np.median(np.where(filtered_mask==2)[1]))
    centroids = (ipsi_cent,contr_cent)
    extremity = ['ipsi','contra']
    return image_to_predict,filtered_mask, centroids, extremity, intensity_dic

# Choose by image
# use this when you want to create a single file .exe
if hasattr(sys, '_MEIPASS') and False:
    base_path = sys._MEIPASS
else:
    base_path = os.getcwd()

# in mac OS, the path is given as follows
if os.name =='posix':
    base_path = os.path.dirname(os.path.abspath(__name__)) # it was __file__ before I changed it due to an error while excusion

if os.path.exists(base_path+"/_internal"):
    base_path = base_path+"/_internal"
print('Base path is',base_path)
#data = pd.read_excel(base_path+'/pt_info.xlsx')
#image_dir = r"\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\opg paper\Segmagic\all_images\CRPS007_1.tiff"
#image_dir = r"\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\Documents\opg paper\Segmagic\all_images\CRPS007_2.png"
#model_type = 'hand'
#side_info = 'left'
#fast_mode = True
#find_intensity(image_dir,side_info, model_type,fast_mode)

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
    def __init__(self, log_dpi):
        super().__init__()
        self.setWindowIcon(QIcon(base_path+'/logo_icon.ico'))
        self.setWindowTitle("ScintiSeg")

        # change window size depending on screen size
        #sc_width = QDesktopWidget().screenGeometry(-1).width()
        self.MF  = log_dpi/96 # magnification factor

        self.main_font = 'Arial'
        # create the global list of images
        self.images = []
        self.im_ind = -1

        self.fast_mode = False # set fast mode to False by default
        # Create main layout
        self.main_layout = QHBoxLayout()

        self.right_layout = QVBoxLayout()
        self.left_layout = QVBoxLayout()
        self.output_layout = QVBoxLayout()
        self.toggle_layout = QHBoxLayout()
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
        self.image_layout = QGridLayout()
        self.tableLayout = QGridLayout()
        #self.handfoot_lo.setContentsMargins(30, 0, 0, 0)
        
        self.title_label = QLabel("ScintiSeg", self)
        self.title_label.setWordWrap(True)
        self.title_label.setFont(QFont(self.main_font, int(20)))
        self.title_label.setAlignment(Qt.AlignCenter)

        renderer = QPixmap(base_path+"/logo.png")
        logo_label = QLabel()
        logo_label.setFixedSize(int(75 * self.MF), int(75 * self.MF))
        logo_label.setPixmap(renderer.scaled(logo_label.size(), QtCore.Qt.KeepAspectRatio))

        qm_icon = QPixmap(base_path+"/question.png")

        self.toggle = AnimatedToggle(
            checked_color="#FFB000",  # Custom color when checked
            pulse_checked_color="#44FFB000")  # Custom color during pulse animation
        self.toggle.setFixedSize(self.toggle.sizeHint())

        self.toggle_info = QLabel("  Fast segment.")
        self.toggle_info.setFont(QFont(self.main_font,int(16)))
        toggle_qm_label = QLabel()
        toggle_qm_label.setFixedSize(int(60 * self.MF), int(30 * self.MF))
        toggle_qm_label.setPixmap(qm_icon.scaled(toggle_qm_label.size(), QtCore.Qt.KeepAspectRatio))
        toggle_qm_label.setToolTip('Fast segmentation mode gives results faster, but is less accurate.\n(Experimental feature, use with caution)')



        single_img_label = QLabel("Segmenting a Single Image")
        single_img_label.setFont(QFont(self.main_font, int(18)))

        folder_seg_label = QLabel("Segmenting Multiple Images")
        folder_seg_label.setFont(QFont(self.main_font, int(18)))

        folder_seg_label_add = QLabel("(Hover over the question mark for more info)")
        folder_seg_label_add.setFont(QFont(self.main_font, int(14)))
        folder_seg_label_add.setWordWrap(True)

        tooltip_label = QLabel()
        tooltip_label.setFixedSize(int(80 * self.MF), int(40 * self.MF))
        tooltip_label.setPixmap(qm_icon.scaled(tooltip_label.size(), QtCore.Qt.KeepAspectRatio))
        tip = ("Choose a data set (excel file)\nIt should have three columns (ID, Side and Extremity), containing\ninformation about the images to be segmented.\nThe image naming should follow the following scheme:\n<subID_scintiPhase>.tif (e.g. CRPS006_3.tif)")
        tooltip_label.setToolTip(f'<img src="{base_path+"/tooltip.png"}">')
        #tooltip_label.setToolTipDuration(5000)

        # Crete common font
        font_sub = QFont(self.main_font, int(14))

        # Styles
        style_allround = "border-radius: 5px; background-color: lightgrey"
        style_right_round = "border-bottom-right-radius: 10px; border-top-right-radius: 10px; border-style: solid; border-width: 1px; border-color: black"
        style_left_round = "border-bottom-left-radius: 10px; border-top-left-radius: 10px; background-color: lightgrey;"
        
        label_txt = "For single-image analysis\nEither drag and drop an image here,\nor use the 'Choose Image' button,\nand press 'Segment Image'"
        self.label1 = image_label(label_txt,self)
        self.label1.setAlignment(Qt.AlignCenter)
        #self.label1.setAcceptDrops(True)
        self.label1.setFixedSize(int(400*self.MF), int(400*self.MF))
        self.label1.setStyleSheet("background-color: white;border-style: solid; border-width: 1px; border-color: black")
        self.label1.setFont(font_sub)
        # Add navigation buttons 

        self.next_icon = QIcon(base_path + "/next.png")
        self.prev_icon = QIcon(base_path + "/prev.png")
        self.none_icon = QIcon()
        self.next_button = QPushButton()
        self.next_button.setFixedHeight(int(400*self.MF))
        self.next_button.setFixedWidth(int(30*self.MF))
        self.next_button.setStyleSheet("QPushButton {background-color: rgba(255, 255, 255, 0);}")
        self.next_button.setFlat(True)

        self.prev_button = QPushButton()
        self.prev_button.setFixedHeight(int(400*self.MF))
        self.prev_button.setFixedWidth(int(30*self.MF))
        self.prev_button.setStyleSheet("QPushButton {background-color: rgba(255, 255, 255, 0);}")
        self.prev_button.setFlat(True)

        # Create data selection widgets
        self.data_path_line = QLineEdit()
        self.data_path_line.setFixedHeight(int(44*self.MF))
        self.data_path_line.setStyleSheet(style_right_round)
        self.databrowse_button = QPushButton("Choose Metadata")
        self.databrowse_button.setStyleSheet(style_left_round)
        self.databrowse_button.setFont(QFont(self.main_font,11))
        self.databrowse_button.setFixedSize(int(130*self.MF), int(44*self.MF))
        
        self.name_error_label = QLabel()
        self.name_error_label.setWordWrap(True)


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
        self.sciphase_1 = QRadioButton("1")
        self.sciphase_1.setChecked(True)
        self.sciphase_2 = QRadioButton("2")
        self.sciphase_3 = QRadioButton("3")
        #self.sciphase_label.setAlignment(Qt.AlignCenter)
        self.sciphase_label.setFont(font_sub)
        self.sciphase_1.setFont(font_sub)
        self.sciphase_2.setFont(font_sub)
        self.sciphase_3.setFont(font_sub)
        
        self.spacer_label = QLabel("   ")

        # Create image selection widgets
        self.img_path_line = QLineEdit()
        self.img_path_line.setFixedHeight(int(44*self.MF))
        self.img_path_line.setStyleSheet(style_right_round)
        
        self.folder_path_line = QLineEdit()
        self.folder_path_line.setFixedHeight(int(44*self.MF))
        self.folder_path_line.setStyleSheet(style_right_round)

        
        self.folderbrowse_button = QPushButton("Choose Image\nFolder")
        self.folderbrowse_button.setFont(QFont(self.main_font,12))
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



        self.exit_button = QPushButton("")
        self.exit_button.setStyleSheet(style_allround)
        self.exit_button.setIcon(QIcon(exit_dir))
        self.exit_button.setIconSize(QtCore.QSize(int(65*self.MF),int(26*self.MF)))
        self.exit_button.setFixedHeight(int(44*self.MF))
        

        # Now the single-image-exclusive layout
        self.imagebrowse_button = QPushButton("Choose Image")
        self.imagebrowse_button.setStyleSheet(style_left_round)
        self.imagebrowse_button.setFont(QFont(self.main_font,14))
        self.imagebrowse_button.setFixedSize(int(130*self.MF), int(44*self.MF))
        self.segment_folder_button = QPushButton("")
        self.segment_folder_button.setStyleSheet(style_allround)
        self.segment_folder_button.setIcon(QIcon(segment_folder_dir))
        self.segment_folder_button.setIconSize(QtCore.QSize(int(80*self.MF),int(80*self.MF)))
        self.segment_folder_button.setFixedHeight(int(88*self.MF))
        
        self.table_w = int(self.MF*510)
        if os.name =='posix':
            self.table_w = int(self.MF * 700)
        copy_h = int(self.table_w * 15 / 400)
        copy_w = int(copy_h * 400 / 15)
        self.table_w = copy_w

        self.createTable()
        self.table_ex = QLabel('Note: for hand images in phase 3, the results consist of three values: Carpal, MCP and IP joints resepctively')
        self.table_ex.setWordWrap(True)
        self.table_ex.setFont(QFont(self.main_font, int(10)))
        self.table_ex.setAlignment(Qt.AlignCenter)
        self.table_ex.setFixedWidth(self.table_w)


        self.tableWidget.setFixedWidth(self.table_w)
        # Create the copy button
        copy_dir = base_path+"/copy.png"

        self.copy_icon = QIcon(copy_dir)
        self.copy_button = QPushButton()

        self.copy_button.setFixedWidth(int(0.8*copy_w))
        self.copy_button.setFixedHeight(copy_h)
        self.copy_button.setStyleSheet("QPushButton {background-color: transparent;}")
        self.copy_button.setFlat(True)


        # Connect button click event to function
        self.folderbrowse_button.clicked.connect(self.browse_folder)
        self.imagebrowse_button.clicked.connect(self.browse_image)
        self.databrowse_button.clicked.connect(self.browse_data)
        self.segment_img_button.clicked.connect(self.segment_image)
        self.segment_folder_button.clicked.connect(self.segment_folder)
        self.toggle.toggled.connect(self.toggle_changed)
        self.exit_button.clicked.connect(self.close)
        self.next_button.clicked.connect(self.next_image)
        self.prev_button.clicked.connect(self.prev_image)
        self.copy_button.clicked.connect(self.copy_table)



        # Build the layouts
        self.aff_side_lo.addWidget(self.aff_side_label,0,0,1,2)
        self.aff_side_lo.addWidget(self.aff_side_left,1,0,1,1)
        self.aff_side_lo.addWidget(self.aff_side_right,1,1,1,1)

        self.handfoot_lo.addWidget(self.handfoot_label,0,0,1,2)
        self.handfoot_lo.addWidget(self.handfoot_hand,1,0,1,1)
        self.handfoot_lo.addWidget(self.handfoot_foot,1,1,1,1)

        self.sciphase_lo.addWidget(self.sciphase_label,0,0,1,2)
        self.sciphase_lo.addWidget(self.sciphase_1,1,0,1,1)
        self.sciphase_lo.addWidget(self.sciphase_2,1,1,1,1)
        self.sciphase_lo.addWidget(self.sciphase_3,1,2,1,1)


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

        Separador3 = QFrame()
        Separador3.setFrameShape(QFrame.HLine)  # Set the shape to horizontal line
        Separador3.setStyleSheet('color: lightgrey')
        Separador3.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        Separador3.setLineWidth(int(1 * self.MF))

        # Add all widgets to thier respective layouts
        #self.output_layout.addWidget(self.label1)
        #self.output_layout.addWidget(self.tableWidget)

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

        self.folderselection_lo.addWidget(self.segment_folder_button,0,9,2,2)
        self.folderselection_lo.addWidget(self.folderbrowse_button,0,0,1,2)
        self.folderselection_lo.addWidget(self.folder_path_line,0,2,1,6)
        self.folderselection_lo.addWidget(self.spacer_label,0,8,1,1)
        self.folderselection_lo.addWidget(self.databrowse_button,1,0,1,2)
        self.folderselection_lo.addWidget(self.data_path_line,1,2,1,6)
        self.folderselection_lo.addWidget(self.name_error_label,2,0,1,11)
        self.folderselection_lo.setHorizontalSpacing(0)

        self.toggle_layout.addWidget(self.toggle_info)
        self.toggle_layout.addWidget(toggle_qm_label)
        self.toggle_layout.addWidget(self.toggle)
        
        self.finalset_lo.addWidget(self.exit_button)
        self.finalset_lo.addLayout(self.toggle_layout)
        #self.finalset_lo.addWidget(self.spacer_label)
        #self.finalset_lo.addWidget(self.spacer_label)
        #self.finalset_lo.addWidget(self.spacer_label)
        
        
        self.title_lo.addWidget(logo_label)
        self.title_lo.addWidget(self.title_label)

        self.folder_tootlip_lo.addWidget(folder_seg_label)
        self.folder_tootlip_lo.addWidget(tooltip_label)


        self.tableLayout.addWidget(self.tableWidget,0,0,8,10)
        self.tableLayout.addWidget(self.copy_button,7,1,1,8)
        self.tableLayout.addWidget(self.table_ex,8,0,1,10)
        

        # Add all Layouts to the main one
        self.right_layout.addWidget(single_img_label)
        self.right_layout.addLayout(self.image_info_lo)
        self.right_layout.addLayout(self.imageselection_lo)
        self.right_layout.addWidget(Separador3)
        #self.right_layout.addWidget(folder_seg_label)
        self.right_layout.addLayout(self.folder_tootlip_lo)
        self.right_layout.addLayout(self.dataselection_lo)
        self.right_layout.addLayout(self.folderselection_lo)
        self.right_layout.addLayout(self.tableLayout)

        self.image_layout.addWidget(self.next_button,0,0,1,1)
        self.image_layout.addWidget(self.label1,0,0,1,5)
        self.image_layout.addWidget(self.prev_button,0,4,1,1)

        self.left_layout.addLayout(self.title_lo)
        self.left_layout.addLayout(self.image_layout)
        self.left_layout.addLayout(self.finalset_lo)
        
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addWidget(self.spacer_label)
        self.main_layout.addLayout(self.right_layout)
        
        # Set the layout
        self.setLayout(self.main_layout)
        center_point = QDesktopWidget().availableGeometry().center()
        self.move(center_point - self.rect().center())
    def copy_table(self):
        rows = []
        for row in range(self.tableWidget.rowCount()):
            cols = []
            for col in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, col).text()
                cols.append(item)
            rows.append(cols)
        df = pd.DataFrame(rows, columns = ['ID','Limb','Phase','Ipsi','Contra','Ratio'])
        df.drop_duplicates(inplace =True)

        n_phases = len(np.unique(df['Phase']))

        val_cols = ['Limb','Ipsi','Contra','Ratio']
        new_df = df.pivot(index='ID',values = val_cols, columns='Phase')
        colnames = []
        for colname in val_cols:
            for i in range(1,n_phases+1):
                colnames.append(f'{colname}_P{str(i)}')

        new_df.columns = colnames
        new_df['ID'] = new_df.index
        new_df.reset_index(drop = True, inplace=True)

        for item in ['Limb_P2','Limb_P3']:
            if item in new_df.columns:
                new_df.drop(item,axis = 1, inplace=True)

        new_df.rename({'Limb_P1':'Limb'},axis = 1,inplace=True)
        new_df = new_df[['ID']+list(new_df.columns)[0:-1]]
        new_df.to_clipboard(index = False)

    def toggle_changed(self):
        if self.toggle.isChecked():
            self.fast_mode = True
        else:
            self.fast_mode = False

    def button_active(self, button, loc_icon, active = True):
        """make a button active or inactive"""
        if active:
            button.setStyleSheet("QPushButton {background-color: rgba(255, 255, 255, 0);}"
                                 "QPushButton:hover {background-color: rgba(128, 128, 128, 25);}"
                                 "QPushButton:pressed { background-color: rgba(128, 128, 128, 75); }")
            button.setIcon(loc_icon)
            wid = button.frameGeometry().width()
            hig = button.frameGeometry().height()
            button.setIconSize(QtCore.QSize(int(wid), int(hig)))
        else:
            button.setStyleSheet("QPushButton {background-color: rgba(255, 255, 255, 0);}")
            button.setIcon(self.none_icon)
            wid = button.frameGeometry().width()
            hig = button.frameGeometry().height()
            button.setIconSize(QtCore.QSize(int(wid), int(hig)))

    def next_image(self):
        if self.im_ind<(len(self.images)-1):
            self.im_ind+=1
            im = self.images[self.im_ind]
            self.plot_mask(im[0],im[1],im[2],im[3],im[4])

            # make sure the previous button icon appears
            self.button_active(self.prev_button,self.prev_icon, active = True)

        if self.im_ind==(len(self.images)-1):
            # make sure the next icon disappears when you reach the final image
            self.button_active(self.next_button,self.none_icon, active = False)

    def prev_image(self):
        if self.im_ind>0:
            self.im_ind-=1
            im = self.images[self.im_ind]
            self.plot_mask(im[0],im[1],im[2],im[3],im[4])

            self.button_active(self.next_button, self.next_icon, active=True)
        if self.im_ind == 0:
            self.button_active(self.prev_button,self.none_icon, active = False)


        
    def editTable(self, content):
        current_row_count = self.tableWidget.rowCount()
        self.tableWidget.insertRow(current_row_count)
        ipsi = QTableWidgetItem(content['ipsi'])
        ipsi.setTextAlignment(Qt.AlignCenter)
        contra = QTableWidgetItem(content['contra'])
        contra.setTextAlignment(Qt.AlignCenter)
        ratio = QTableWidgetItem(content['ratio'])
        ratio.setTextAlignment(Qt.AlignCenter)

        id = QTableWidgetItem('_'.join(content['id'].split('_')[0:-2]))
        id.setTextAlignment(Qt.AlignCenter)
        limb = QTableWidgetItem(content['id'].split('_')[-1])
        limb.setTextAlignment(Qt.AlignCenter)
        phase = QTableWidgetItem(content['id'].split('_')[-2])
        phase.setTextAlignment(Qt.AlignCenter)
        self.tableWidget.setItem(current_row_count, 0, id)
        self.tableWidget.setItem(current_row_count, 1, limb)
        self.tableWidget.setItem(current_row_count, 2, phase)
        self.tableWidget.setItem(current_row_count, 3, ipsi)
        self.tableWidget.setItem(current_row_count, 4, contra)
        self.tableWidget.setItem(current_row_count, 5, ratio)
  
        
    def createTable(self):
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(6) 
        self.tableWidget.setHorizontalHeaderLabels(["ID","Limb","Phase","Ipsi", "Contra","Ratio"])
        self.tableWidget.setMinimumHeight(int(200*self.MF))
        self.tableWidget.setMinimumWidth(int(450*self.MF))
        #self.tableWidget.horizontalHeader().setStretchLastSection(True)  # Stretch the last section
        #self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Resize mode
        self.tableWidget.setColumnWidth(0, int(2*(self.table_w)/13))
        self.tableWidget.setColumnWidth(1, int((self.table_w)/13)+5)
        self.tableWidget.setColumnWidth(2, int((self.table_w)/13))
        self.tableWidget.setColumnWidth(3, int(3*(self.table_w)/13))
        self.tableWidget.setColumnWidth(4, int(3*(self.table_w)/13))
        self.tableWidget.setColumnWidth(5, int(3*(self.table_w)/13)-45)





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
        file_path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "TIF images (*.tif);PNG images (*.png);JPG images (*.jpg);All Files (*)", options=options)
        if file_path:
            self.img_path_line.setText(file_path)
            self.load_image()

    def load_image(self):
        image_path = self.img_path_line.text()
        pixmap = QPixmap(image_path)
        # Set the pixmap to the first QLabel
        self.label1.setPixmap(pixmap.scaled(self.label1.size(), QtCore.Qt.KeepAspectRatio))

    def segment_folder(self):
        self.name_error_label.setText('')
        data = pd.read_excel(self.data_path_line.text())
        data_accepted = np.all([item in data.columns for item in ['ID','Side','Extremity']]) # Check if all columns are found in the data
        if data_accepted:
            folder_dir = self.folder_path_line.text()
            file_names = [item.split("/")[-1].split("\\")[-1] for item in glob.glob(folder_dir+"/*")]

            # Only proceed with the directories that follow the naming convention
            accepted_filenames = []
            for fn in file_names:
                splitname = ".".join(fn.split(".")[0:-1]).split("_")
                if len(splitname)==2:
                        if splitname[1].isdigit():
                            accepted_filenames.append(fn)
            wrong_filenames = [fn for fn in file_names if fn not in accepted_filenames]
            err_text = ""
            if len(wrong_filenames)>0:
                err_text += f"(See tooltip) The following filenames do not match the naming scheme: {wrong_filenames}"
        
            ids = [item.split("_")[0] for item in accepted_filenames]
            ids_notindata = [id for id in ids if not (id in data['ID'].values)]
            accepted_vals = ['upper','lower','right','left','Upper','Lower','Right','Left']
            ids_withoutinfo = [id for id in ids if not np.all(data.loc[data['ID']==id,['Side','Extremity']].isin(accepted_vals))]
            wrong_ids = ids_notindata+ids_withoutinfo
            if len(wrong_ids)>0:
                err_text += f" ..... The following IDs have no data in the data frame: {wrong_ids}"
            if len(err_text)>0:
                self.name_error_label.setText(f'<font color="darkred">{err_text}</font>')

            accepted_filenames = [item for item in accepted_filenames if item.split("_")[0] not in wrong_ids]
            if len(accepted_filenames)>0:
                for fn in accepted_filenames:
                    image_dir = f"{folder_dir}/{fn}"
                    splitname = ".".join(fn.split(".")[0:-1]).split("_")
                    id,phase = splitname
                    extremity = data.loc[data['ID']==id,'Extremity'].values[0].lower()
                    model_type = 'hand' if (('up' in extremity) or ('hand' in extremity)) else 'foot'

                    # hand_p3 is a special model for the hands
                    model_type = 'hand_p3'if model_type+phase == 'hand3' else model_type
                    
                    affected_limb = 'hand' if model_type.startswith('hand') else 'foot'

                    side_info = data.loc[data['ID']==id,'Side'].values[0].lower()
                    
                    ID = image_dir.split("/")[-1].split("\\")[-1]
                    image_to_predict,filtered_mask, centroids, region_afctd_extr, intensity_dic = find_intensity(image_dir,side_info, model_type, self.fast_mode)
                    intensity_dic['id'] += f'_{affected_limb}'
                    self.images.append([image_to_predict,filtered_mask,centroids, region_afctd_extr,ID])
                    self.im_ind+=1
                    self.plot_mask(image_to_predict, filtered_mask,centroids, region_afctd_extr,image_dir.split("/")[-1].split("\\")[-1])
                    self.editTable(intensity_dic)

                    # make sure that the previous-button icon appears
                    if self.im_ind > 0:
                        self.button_active(self.prev_button, self.prev_icon, active=True)
                    self.button_active(self.copy_button, self.copy_icon, active=True)
                    self.button_active(self.next_button, self.none_icon, active=False)

        else:
            self.name_error_label.setText(f'<font color="darkred">the chosen data frame does not have the needed columns (ID, Side, Extremity)</font>')
    def segment_image(self):
        image_dir = self.img_path_line.text()
        if os.path.exists(image_dir):
            side_info = "left" if self.aff_side_left.isChecked() else "right"
            model_type = "hand" if self.handfoot_hand.isChecked() else "foot"
            phase = "1"
            if self.sciphase_2.isChecked():
                phase = "2"
            if self.sciphase_3.isChecked():
                phase = "3"
            
            model_type = 'hand_p3' if model_type+phase == 'hand3' else model_type
            affected_limb = 'hand' if model_type.startswith('hand') else 'foot'
            image_to_predict,filtered_mask, centroids, region_afctd_extr, intensity_dic = find_intensity(image_dir,side_info, model_type, self.fast_mode)
            intensity_dic['id'] += f'_{phase}_{affected_limb}'

            ID = image_dir.split("/")[-1].split("\\")[-1]
            self.images.append([image_to_predict,filtered_mask,centroids, region_afctd_extr,ID])
            self.im_ind+=1
            self.plot_mask(image_to_predict, filtered_mask,centroids, region_afctd_extr, ID)
            self.editTable(intensity_dic)

            # make sure that the previous-button icon appears
            if self.im_ind>0:
                self.button_active(self.prev_button, self.prev_icon, active=True)
            self.button_active(self.copy_button, self.copy_icon, active=True)
            self.button_active(self.next_button, self.none_icon, active=False)
    def plot_mask(self,image_to_predict, filtered_mask,centroids, region_afctd_extr,ID):
        color_list = ['steelblue', 'indianred','olivedrab','darkgoldenrod','darkmagenta','grey','palevioletred','sienna','beige','coral']

        cmap = plt.cm.colors.ListedColormap(['white']+color_list[0:len(centroids)])
        plt.close()
        fig = plt.subplots(figsize = (5,5))
        plt.imshow(image_to_predict[0,:,:],cmap='gray')

        plt.imshow(filtered_mask, cmap=cmap, interpolation='nearest', alpha = 0.3)
        # Add text
        for centroid,l_text in zip(centroids,region_afctd_extr):
            plt.text(centroid[1],centroid[0],l_text, ha='center', font = self.main_font, size = 26)#path_effects=[patheffects.withStroke(linewidth=2, foreground='white')]
        plt.text(200,395,ID, ha='center', font = self.main_font, size = 20)#path_effects=[patheffects.withStroke(linewidth=2, foreground='white')]

        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        plt.savefig(base_path+"/temp_file.png")
        plt.close()
        pixmap_regions = QPixmap(base_path+"/temp_file.png")
        self.label1.setPixmap(pixmap_regions.scaled(self.label1.size(), QtCore.Qt.KeepAspectRatio))
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
    primary_screen = app.primaryScreen()
    logical_dpi = primary_screen.logicalDotsPerInch()
    window = MyWindow(logical_dpi)
    window.show()
    sys.exit(app.exec_())

