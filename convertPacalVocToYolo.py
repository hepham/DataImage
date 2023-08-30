import os
import xml.etree.ElementTree as ET
import random
import shutil
from sklearn.model_selection import train_test_split
def convert_voc_to_yolo(xml_folder, yolo_folder, classes):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            yolo_file_path = os.path.join(yolo_folder, xml_file.replace('.xml', '.txt'))
            with open(yolo_file_path, 'w') as yolo_file:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in classes:
                        continue

                    class_id = classes.index(class_name)
                    bbox = obj.find('bndbox')
                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))


                    width = xmax - xmin
                    height = ymax - ymin
                    x_center = (xmin + xmax) / 2
                    y_center = (ymin + ymax) / 2

                    yolo_line = f"{class_id} {x_center/640} {y_center/640} {width/640} {height/640}\n"
                    yolo_file.write(yolo_line)

# Define class names (in the same order as they appear in the YOLO model's class list)
class_names = ['1', '2', '3', '36', '4', '5', '6', '7', '8', '9', 'a', 'acong', 'and', 'b', 'c', 'd', 'dolar', 'e', 'f', 'h', 'j', 'k', 'l', 'n', 'p', 'per', 'q', 'r', 's', 't', 'thang', 'u', 'v', 'w', 'x', 'y', 'z']




def split_dataset(input_folder, train_folder, test_folder, valid_folder, split_percent=(0.8, 0.1, 0.1)):
    file_list = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    train_files, test_valid_files = train_test_split(file_list, test_size=(split_percent[1] + split_percent[2]), random_state=42)
    test_files, valid_files = train_test_split(test_valid_files, test_size=split_percent[2] / (split_percent[1] + split_percent[2]), random_state=42)
    for folder in [train_folder, test_folder, valid_folder]:
        os.makedirs(folder, exist_ok=True)
        os.makedirs(folder+"/images", exist_ok=True)
        os.makedirs(folder+"/labels", exist_ok=True)

    for file in train_files:
        image_path = os.path.join(input_folder, file)
        txt_file = file.replace('.png', '.txt')
        txt_path = os.path.join(input_folder, txt_file)
        shutil.copy(image_path, os.path.join(train_folder+"/images", file))
        shutil.copy(txt_path, os.path.join(train_folder+"/labels", txt_file))
        
    for file in test_files:
        image_path = os.path.join(input_folder, file)
        txt_file = file.replace('.png', '.txt')
        txt_path = os.path.join(input_folder, txt_file)
        shutil.copy(image_path, os.path.join(test_folder+"/images", file))
        shutil.copy(txt_path, os.path.join(test_folder+"/labels", txt_file))

    for file in valid_files:
        image_path = os.path.join(input_folder, file)
        txt_file = file.replace('.png', '.txt')
        txt_path = os.path.join(input_folder, txt_file)
        shutil.copy(image_path, os.path.join(valid_folder+"/images", file))
        shutil.copy(txt_path, os.path.join(valid_folder+"/labels", txt_file))

# Paths to XML folder and YOLO output folder
xml_folder_path = './convertxml'
yolo_folder_path = './data'
# Define paths
input_folder_path = './data'
train_folder_path = './dataset/train'
test_folder_path = './dataset/test'
valid_folder_path = './dataset/valid'
# Convert
convert_voc_to_yolo(xml_folder_path, yolo_folder_path, class_names)
# Define split percentages
split_percentages = (0.8, 0.1, 0.1)

# Split dataset
split_dataset(input_folder_path, train_folder_path, test_folder_path, valid_folder_path, split_percentages)



