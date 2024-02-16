import streamlit as st
from PIL import Image
from io import BytesIO
import os, random, cv2, json
import numpy as np

st.set_page_config(layout="wide", page_title="Visualize Image")

st.write("## Visualize image")

ROOT_PATH = '/data/ephemeral/home/datasets/train/DCM'
trapezium_log = './EDA/trapezium.txt'
pisiform_log = './EDA/pisiform.txt'

trapezium_list = open(trapezium_log, 'r').readlines()
pisiform_list = open(pisiform_log, 'r').readlines()

def vis_image(img_path, score):
    # image = Image.open(os.path.join(ROOT_PATH, img_path))
    image = cv2.imread(os.path.join(ROOT_PATH, img_path))
    col2.write(f"path : {img_path}, score : {score}")
    col2.image(image)

def vis_gt(img_path, score, option):
    image_path = os.path.join(ROOT_PATH, img_path)
    json_path = image_path.replace('DCM', 'outputs_json').replace('.png', '.json')

    with open(json_path, 'r') as r:
        json_data = json.load(r)
   
    image = cv2.imread(image_path)
    annos = json_data['annotations']
    for ann in annos:
        if ann['label'] == option:
            cv2.fillPoly(image, [np.array(ann['points'])], (0,255,0))
    col2.write(f"path : {img_path}, score : {score}")
    col2.image(image)

col1, col2 = st.columns(2)

option = col1.selectbox(
   "choose class",
   ("Trapezium", "Pisiform"),
)

if option == 'Trapezium':
    txt_infos = trapezium_list
elif option == 'Pisiform':
    txt_infos = pisiform_list

txt_infos = [txt_info.split() for txt_info in txt_infos]
txt_infos.sort(key=lambda x: (x[1], x[0]))
cols = col1.columns(4)
cols[0].write('#### path')
cols[1].write('#### score')
cols[2].write('#### vis_img')
cols[3].write('#### vis_gt')
for p, s in txt_infos:
    cols = col1.columns(4)
    cols[0].write(f'{p}')
    cols[1].write(f'{s}')
    cols[2].button('Click', key=p, type='primary', on_click=vis_image, args=(p, s,))
    cols[3].button('Draw GT', key=f'{p}_gt', type='primary', on_click=vis_gt, args=(p, s, option))
