import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
import keras
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from io import BytesIO

your_path = ""

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def load_model():
	model = tf.keras.models.load_model(your_path + '\resnet101_overfit.hdf5')
	return model


def predict_class(image, model):
# 	image = tf.cast(image, tf.float32)
	image = np.resize(image, (224,224))
# 	image_1 = image
	image = np.dstack((image,image,image))
# 	image_2 = image
# 	cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	image = np.expand_dims(image, axis = 0)
# 	image_3 = image   


	prediction = model.predict(image)

	return prediction

def preprocessing_uploader(file, model):
    bytes_data = file.read()
    inputShape = (224, 224)
    image = Image.open(BytesIO(bytes_data))
    image = image.convert("RGB")
    image = image.resize(inputShape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    prediction = model.predict(image) 
    return prediction
app_mode = st.sidebar.selectbox('Chọn trang',['Thông tin chung','Thống kê về dữ liệu huấn luyện','Ứng dụng chẩn đoán']) #two pages
if app_mode=='Thông tin chung':
    st.title('Giới thiệu về thành viên')
    st.markdown("""
    <style>
    .big-font {
    font-size:35px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .name {
    font-size:25px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font"> Học sinh thực hiện </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> I. Trần Mạnh Dũng - 11A1 </p>', unsafe_allow_html=True)
    dung_ava = Image.open(your_path + '\member\Dung.jpg')
    st.image(dung_ava)
    st.markdown('<p class="name"> II. Lê Vũ Anh Tin - 8A2 </p>', unsafe_allow_html=True)
    tin_ava = Image.open(your_path + r'\member\Tin.jpg')
    st.image(tin_ava)
    
    st.markdown('<p class="big-font"> Giáo viên hướng dẫn đề tài </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> Lê Thúy Phương Như - Giáo viên Sinh Học </p>', unsafe_allow_html=True)
    Nhu_ava = Image.open(your_path + r'\member\GVHD_Nhu.jpg')
    st.image(Nhu_ava)
elif app_mode=='Thống kê về dữ liệu huấn luyện': 
    st.title('Thống kê tổng quan về tập dữ liệu')
    
    st.markdown("""
    <style>
    .big-font {
    font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font"> I. Thông tin về tập dữ liệu </p>', unsafe_allow_html=True)
    st.caption('Tập dữ liệu COVID-QU-EX được các nhà nghiên cứu của Đại Học Qatar (Qatar Univerersity) thu thập, làm sạch và chuẩn bị. Tập dữ liệu bao gồm 33920 ảnh X-quang lồng ngực, trong đó bao gồm 11956 ảnh có ghi nhận mắc covid-19, 11263 ảnh có ghi nhận mắc viêm phổi không do covid và 10701 ảnh bình thường. ')
    st.caption('Nội dung nghiên cứu khoa học và ứng dụng của nhóm được thiết kế dựa trên việc huấn luyện nhóm dữ liệu Lung Segmentation Data. Dữ liệu đã được tiền xử lý và thay đổi kích thước về 256 x 256. Thông tin chi tiết của tập dữ liệu có thể tìm được ở dưới đây: ')
    st.caption('*"https://www.kaggle.com/datasets/anasmohammedtahir/covidqu"*')
    covid_dataset = Image.open(your_path + r'\stat_image\covid_dataset.png')
    st.image(covid_dataset)
    #Vẽ sample ảnh
    st.text('1) Một vài mẫu của ảnh x-quang lồng ngực mắc covid-19.')
    covid_sample = Image.open(your_path + r'\stat_image\covid_sample.png')
    st.image(covid_sample)
    
    st.text('2) Một vài mẫu của ảnh x-quang lồng ngực mắc viêm phổi thông thường.')
    non_covid_sample = Image.open(your_path + r'\stat_image\non_covid_sample.png')
    st.image(non_covid_sample)
    
    st.text('3) Một vài mẫu của ảnh x-quang lồng ngực khỏe mạnh.')
    normal_sample = Image.open(your_path + r'\stat_image\normal_sample.png')
    st.image(normal_sample)
    
    st.text('4) Vùng quan trọng được mô hình học máy chú ý.')
    gradcam = Image.open(your_path + r'\stat_image\gradcam.png')
    st.image(gradcam)
    
    #Vẽ thống kê tập dữ liệu
    st.markdown('<p class="big-font"> II. Thống kê về tập dữ liệu </p>', unsafe_allow_html=True)
    st.caption('Nhìn chung, dữ liệu tương đối cân bằng ở 3 lớp, trên cả tập huấn luyện và tập kiểm thử với lần lượt là ')
    st.text('1) Biểu đồ cột so sánh số lượng dữ liệu tập huấn luyện (Train dataset)')
    train_info = Image.open(your_path + r'\stat_image\train_info.png')
    st.image(train_info)
    st.text('2) Biểu đồ cột so sánh số lượng dữ liệu tập kiểm thử (Validation dataset)')
    valid_info = Image.open(your_path + r'\stat_image\valid_info.png')
    st.image(valid_info)
    st.text('3) Biểu đồ tròn so sánh phần trăm dữ liệu tập huấn luyện (Train dataset)')
    train_pie = Image.open(your_path + r'\stat_image\train_pie.png')
    st.image(train_pie)
    st.text('4) Biểu đồ tròn so sánh phần trăm dữ liệu tập kiểm thử (Validation dataset)')
    valid_pie = Image.open(your_path + r'\stat_image\valid_pie.png')
    st.image(valid_pie)
elif app_mode=='Ứng dụng chẩn đoán':
    model = load_model()
    st.title('Ứng dụng chẩn đoán bệnh trong ảnh X-quang lồng ngực')

    file = st.file_uploader("Bạn vui lòng nhập ảnh x-quang lồng ngực để phân loại ở đây", type=["jpg", "png"])
# 

    if file is None:
        st.text('Đang chờ tải lên....')

    else:
        slot = st.empty()
        slot.text('Hệ thống đang thực thi chẩn đoán....')
        
        pred = preprocessing_uploader(file, model)
        test_image = Image.open(file)
        st.image(test_image, caption="Ảnh đầu vào", width = 400)
        class_names = ['covid', 'non-covid','normal']

        result = class_names[np.argmax(pred)]
        
        if str(result) == 'covid':
            statement = str('Chẩn đoán của mô hình học máy: **Bệnh nhân mắc Covid-19.**')
            st.error(statement)
        elif str(result) == 'non-covid':
            statement = str('Chẩn đoán của mô hình học máy: **Bệnh nhân mắc viêm phổi không do virus Covid-19 gây ra.**')
            st.warning(statement)
        elif str(result) == 'normal':
            statement = str('Chẩn đoán của mô hình học máy: **Không có dấu hiệu bệnh viêm phổi.**')
            st.success(statement)
        slot.success('Hoàn tất!')

#         st.success(output)
     
        #Plot bar chart
        bar_frame = pd.DataFrame({'Xác suất dự đoán': [pred[0,0] *100, pred[0,1]*100, pred[0,2]*100], 
                                   'Loại chẩn đoán': ["Covid-19", "Viêm phổi khác", "Bình thường"]
                                 })
        bar_chart = alt.Chart(bar_frame).mark_bar().encode(y = 'Xác suất dự đoán', x = 'Loại chẩn đoán' )
        st.altair_chart(bar_chart, use_container_width = True)
        #Note
        st.write('- **Xác suất bệnh nhân mắc covid-19 là**: *{}%*'.format(round(pred[0,0] *100,2)))
        st.write('- **Xác suất bệnh nhân mắc viêm phổi khác là**: *{}%*'.format(round(pred[0,1] *100,2)))
        st.write('- **Xác suất bệnh nhân khỏe mạnh là**: *{}%*'.format(round(pred[0,2] *100,2)))
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
