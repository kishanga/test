import streamlit as st
import cv2
import torch
from utils.hubconf import custom
from utils.plots import plot_one_box
import numpy as np
import tempfile
from PIL import ImageColor, Image
import time
from collections import Counter
import json
import psutil
import subprocess
import pandas as pd


def get_gpu_memory():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory[0]

def color_picker_fn(classname, key):
    color_picke = st.sidebar.color_picker(f'Bounding Box Color For {classname}:', '#ff0003', key=key)#00f900#ff0003
    color_rgb_list = list(ImageColor.getcolor(str(color_picke), "RGB"))
    color = [color_rgb_list[2], color_rgb_list[1], color_rgb_list[0]]
    return color

p_time = 0

st.title('Cockroach Detection Dashboard')
sample_img = 'https://raw.githubusercontent.com/weizhe38/GA-Project-Weizhe-/main/Capstone/images_videos/S-WALL-E-2008-06.jpg'
#st.image(sample_img)

#FRAME_WINDOW = st.image(sample_img, channels='BGR') #sample_img, channels='BGR'
st.sidebar.title('Settings')

# path to model
path_model_file = st.sidebar.text_input(
    'path to YOLOv7 Model:',
    'eg: dir/yolov7.pt'
) 
#"G:\My Drive\GACapstone\Streamlit_1\sest.pt"

# Class txt
path_to_class_txt = ["Non-Cockroach", "Cockroach"]
## st.sidebar.file_uploader( 'Class.txt:', type=['txt'])

if path_to_class_txt is not None:
    
    # Checkbox for image/video selection
    st.sidebar.markdown('---')
    options = st.sidebar.radio(
        'Detection Options:', ('Image', 'Video'), index=0)
    ##    'Options:', ('Webcam', 'Image', 'Video', 'RTSP'), index=1)
    st.sidebar.markdown('---')

    # Checkbox for PU selection
    gpu_option = st.sidebar.radio(
        'PU Options:', ('CPU', 'GPU'), index = 0)
    
    if not torch.cuda.is_available():
        st.sidebar.warning('CUDA Not Available, So choose CPU', icon="⚠️")
    else:
        st.sidebar.success(
            'GPU is Available on this Device, Choose GPU for the best performance',
            icon="✅"
        )
    st.sidebar.markdown('---')

    # Sliderbar for confidence level
    confidence = st.sidebar.slider(
        'Detection Confidence:' , min_value=0.01, max_value=1.0, value=0.25)
    st.sidebar.markdown('---')

    # Draw thickness
    draw_thick = st.sidebar.slider(
        'Bounding Box Line Thickness:', min_value=1,
        max_value=20, value=3
    )
    st.sidebar.markdown('---')
    # Read class.txt
    ## bytes_data = path_to_class_txt.getvalue()
    class_labels = path_to_class_txt
    color_pick_list = []

    for i in range(len(class_labels)):
        classname = class_labels[i]
        color = color_picker_fn(classname, i)
        color_pick_list.append(color)
    
    st.sidebar.markdown('---')

    # Image Detection
    if options == 'Image':

        st.subheader('Image Detection Guidelines')
        st.caption("""
- Image detection dashboard supports "png", "jpg", "jpeg" image filetypes.  \n
- Recommended to use images with better contrast.  \n 
- How to use confidence level setting:
    - Lower confidence level: Increases number of objects detected, reduces classification accuracy of detected objects.
    - Higher confidence level: Reduces number of objects detected, improves classification accuracy of detected objects.
    """)

        col1, col2 = st.columns(2)
        
        with col1:
            st.caption("Image Input Example")
            input_img = 'https://raw.githubusercontent.com/weizhe38/GA-Project-Weizhe-/main/Capstone/images_videos/test_image.png'
            st.image(input_img, channels = 'BGR')
            
        with col2:
            st.caption("Image Output Example")
            output_img = 'https://raw.githubusercontent.com/weizhe38/GA-Project-Weizhe-/main/Capstone/images_videos/test_image_bb.png'
            st.image(output_img, channels = 'BGR')        

        upload_img_file = st.file_uploader(
            'Upload Image', type=['jpg', 'jpeg', 'png'])
            
        st.subheader('Detection Output')
        FRAME_WINDOW = st.image(sample_img, channels='BGR') #sample_img, channels='BGR'

        if upload_img_file is not None:
            pred = st.button('Detect!')
            file_bytes = np.asarray(
                bytearray(upload_img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            #img_1 = cv2.imdecode(file_bytes, 1)
            FRAME_WINDOW.image(img, channels='BGR')

            if pred:
                if gpu_option == 'CPU':
                    model = custom(path_or_model=path_model_file)
                if gpu_option == 'GPU':
                    model = custom(path_or_model=path_model_file, gpu=True)
                
                bbox_list = []
                current_no_class = []
                results = model(img)
                
                # Bounding Box
                box = results.pandas().xyxy[0]
                class_list = box['class'].to_list()

                for i in box.index:
                    xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                        int(box['ymax'][i]), box['confidence'][i]
                    if conf > confidence:
                        bbox_list.append([xmin, ymin, xmax, ymax])
                if len(bbox_list) != 0:
                    for bbox, id in zip(bbox_list, class_list):
                        plot_one_box(bbox, img, label=class_labels[id],
                                    color=color_pick_list[id], line_thickness=draw_thick)
                        current_no_class.append([class_labels[id]])
                FRAME_WINDOW.image(img, channels='BGR')
                
                # Current number of classes
                class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                class_fq = json.dumps(class_fq, indent = 4)
                class_fq = json.loads(class_fq)
                df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Count'])
                df_fq = df_fq.style.hide_index()

                # Updating detection results
                with st.container():
                    st.markdown("<h3>Detected count of objects in current image</h3>", unsafe_allow_html=True)
                    st.write(df_fq.to_html(),unsafe_allow_html=True) # removes index column
                    #st.dataframe(df_fq,use_container_width=True)

    # Video
    if options == 'Video':
        
        # subheader and Guidelines Caption
        st.subheader('Video Detection Guidelines')
        st.caption("""
- Video detection dashboard supports "mp4" and "mp3" image filetypes.  \n
- How to use confidence level setting:
    - Lower confidence level: Increases number of objects detected, reduces classification accuracy of detected objects.
    - Higher confidence level: Reduces number of objects detected, improves classification accuracy of detected objects.
    """)

        # Upload video input box
        upload_video_file = st.file_uploader(
        'Upload Video', type=['mp4','mp3'])

        # Video Output Sample
        DEMO_VIDEO = 'https://raw.githubusercontent.com/weizhe38/GA-Project-Weizhe-/main/Capstone/images_videos/test_video_0.5_1.mp4'
        
        if upload_video_file is None:
            #tfile = tempfile.NamedTemporaryFile(delete = False)
            #tfile.name = DEMO_VIDEO
            #dem_vid = open(tfile.name, 'rb')
            #demo_bytes = dem_vid.read()

            st.caption("Video Output Example")
            FRAME_WINDOW = st.video(DEMO_VIDEO)        

        # After uploading video for analysis 
        if upload_video_file is not None:
            
            pred = st.button('Detect!')
            
            # Model
            if gpu_option == 'CPU':
                model = custom(path_or_model=path_model_file)
            if gpu_option == 'GPU':
                model = custom(path_or_model=path_model_file, gpu=True)

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(upload_video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            demo_bytes = open(tfile.name,'rb').read()
            st.subheader("Detection Output")
            FRAME_WINDOW = st.video(demo_bytes)

            if pred:
                FRAME_WINDOW.image([])
                stframe1 = st.empty()
                stframe2 = st.empty()
                stframe3 = st.empty()
                while True:
                    success, img = cap.read()
                    if not success:
                        st.error(
                            'Video file NOT working\n \
                            Check Video path or file properly!!',
                            icon="🚨"
                        )
                        break
                    current_no_class = []
                    bbox_list = []
                    results = model(img)
                    # Bounding Box
                    box = results.pandas().xyxy[0]
                    class_list = box['class'].to_list()

                    for i in box.index:
                        xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                            int(box['ymax'][i]), box['confidence'][i]
                        if conf > confidence:
                            bbox_list.append([xmin, ymin, xmax, ymax])
                    if len(bbox_list) != 0:
                        for bbox, id in zip(bbox_list, class_list):
                            plot_one_box(bbox, img, label=class_labels[id],
                                         color=color_pick_list[id], line_thickness=draw_thick)
                            current_no_class.append([class_labels[id]])
                    FRAME_WINDOW.image(img, channels='BGR')

                    # FPS
                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                    
                    # Current number of classes
                    class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                    class_fq = json.dumps(class_fq, indent = 4)
                    class_fq = json.loads(class_fq)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                    
                    # Updating Inference results
                    with stframe1.container():
                        st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                        if round(fps, 4)>1:
                            st.markdown(f"<h4 style='color:green;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h4 style='color:red;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
                    
                    with stframe2.container():
                        st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
                        st.dataframe(df_fq, use_container_width=True)

                    with stframe3.container():
                        st.markdown("<h2>System Statistics</h2>", unsafe_allow_html=True)
                        js1, js2, js3 = st.columns(3)                       

                        # Updating System stats
                        with js1:
                            st.markdown("<h4>Memory usage</h4>", unsafe_allow_html=True)
                            mem_use = psutil.virtual_memory()[2]
                            if mem_use > 50:
                                js1_text = st.markdown(f"<h5 style='color:red;'>{mem_use}%</h5>", unsafe_allow_html=True)
                            else:
                                js1_text = st.markdown(f"<h5 style='color:green;'>{mem_use}%</h5>", unsafe_allow_html=True)

                        with js2:
                            st.markdown("<h4>CPU Usage</h4>", unsafe_allow_html=True)
                            cpu_use = psutil.cpu_percent()
                            if mem_use > 50:
                                js2_text = st.markdown(f"<h5 style='color:red;'>{cpu_use}%</h5>", unsafe_allow_html=True)
                            else:
                                js2_text = st.markdown(f"<h5 style='color:green;'>{cpu_use}%</h5>", unsafe_allow_html=True)

                        with js3:
                            st.markdown("<h4>GPU Memory Usage</h4>", unsafe_allow_html=True)  
                            try:
                                js3_text = st.markdown(f'<h5>{get_gpu_memory()} MB</h5>', unsafe_allow_html=True)
                            except:
                                js3_text = st.markdown('<h5>NA</h5>', unsafe_allow_html=True)