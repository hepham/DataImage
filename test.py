import os
import base64
from PIL import Image
from io import BytesIO
import xml.etree.ElementTree as ET
import sys
import time
import io
from roboflow import Roboflow
from flask import Flask, request
from flask import  jsonify
import requests
from PIL import Image
from io import BytesIO
from roboflow import Roboflow
count=0;
def create_xml_annotation(file_info, objects):
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = file_info["folder"]

    filename = ET.SubElement(annotation, "filename")
    filename.text = file_info["filename"]

    path = ET.SubElement(annotation, "path")
    path.text = file_info["path"]

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = file_info["database"]

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(file_info["width"])
    height = ET.SubElement(size, "height")
    height.text = str(file_info["height"])
    depth = ET.SubElement(size, "depth")
    depth.text = str(file_info["depth"])

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = str(file_info["segmented"])

    for obj in objects:
        object_element = ET.SubElement(annotation, "object")
        name = ET.SubElement(object_element, "name")
        name.text = obj["label"]
        pose=ET.SubElement(object_element,"pose")
        pose.text="Unspecified"
        truncated=ET.SubElement(object_element,"truncated")
        truncated.text="0"
        difficult=ET.SubElement(object_element,"difficult")
        difficult.text="0"
        occluded=ET.SubElement(object_element,"occluded")
        occluded.text="0"
        bndbox = ET.SubElement(object_element, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(obj["xmin"])
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(obj["xmax"])
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(obj["ymin"])
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(obj["ymax"])

    tree = ET.ElementTree(annotation)
    return tree

total=100
def percentage_loading(index, delay):
    global total
    progress = index / total
    bar_length = 30
    block = int(round(bar_length * progress))
    percent = round(progress * 100, 2)  
    bar = "=" * block + "-" * (bar_length - block)
    # loading_animation()
    sys.stdout.write(f"\r[{bar}] {percent:.2f}%")
    sys.stdout.flush()
    # time.sleep(delay)

# total_iterations = 50  # Total number of iterations for the loading animation
# delay = 0.1            # Delay in seconds between iterations

# percentage_loading(total_iterations, delay)
# print("\nDone!")
def convert_base64_to_images_in_folder(folder_path):
    global total
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    total=len(txt_files)
    print(total)
    count=0
    for txt_file in txt_files:
        txt_path = os.path.join(folder_path, txt_file)
        count+=1
        with open(txt_path, 'r') as file:
            base64_string = file.read()

        # Convert the base64 string to image
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))

        # Process the image (replace this with your actual processing)
        newSize=(640,640)
        processed_image = image.resize(newSize)

        # Save the processed image
        processed_image.save(os.path.join(folder_path, f"processed_{txt_file.replace('.txt', '.png')}"))
        percentage_loading(index=count,delay=0.1)
        time.sleep(0.1)

folder_path = './test'  # Replace with the actual path to your folder

# Get a list of text files and process them
# convert_base64_to_images_in_folder(folder_path)
def convertResponseToXml(folder_path):
    global total
    rf = Roboflow(api_key="QzD9J2BujSYOeQhsbmx3")
    project = rf.workspace().project("capchatest-21-5")
    model = project.version(4).model
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    total=len(image_files)
    count=0
    for image_file in image_files:
    # infer on a local image
        data=model.predict(folder_path+"/"+image_file, confidence=40, overlap=30).json()
        objects = data['predictions']
        objectWrite=[]
        for bounding_box in objects:
            # print("bounding:"+ str(bounding_box))
            label=bounding_box['class']
            x0 = bounding_box['x'] - bounding_box['width'] / 2
            x1 = bounding_box['x'] + bounding_box['width'] / 2
            y0 = bounding_box['y'] - bounding_box['height'] / 2
            y1 = bounding_box['y'] + bounding_box['height'] / 2
            new_object={"label":label,"xmin":x0,"xmax":x1,"ymin":y0,"ymax":y1}
            # print(new_object)
            objectWrite.append(new_object)
        file_info = {
            "folder": "",
            "filename":image_file ,
            "path":str(image_file),
            "database": "roboflow.com",
            "width": 640,
            "height": 640,
            "depth": 3,
            "segmented": 0
        }
        count+=1
        xml_tree = create_xml_annotation(file_info, objects=objectWrite)
        xml_tree.write(os.path.join(folder_path, f"{image_file.replace('.png', '.xml')}"))
        percentage_loading(index=count,delay=0.1)
convertResponseToXml(folder_path=folder_path)