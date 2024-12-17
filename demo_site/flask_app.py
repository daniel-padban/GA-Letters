
import PIL
import PIL.Image
import PIL.ImageOps
from flask import Flask, flash, request, redirect, session, url_for
import flask
from werkzeug.utils import secure_filename
import os
import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as tvF
import numpy as np
import json
from modelC3 import CNN

app = Flask(__name__)
app.secret_key = 'A_SECRET1099'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  

def read_json(config_path): #load config dictionary
    with open(config_path) as conf_file:
        config_dict = json.load(conf_file)
        return config_dict 
alphabet_dict = read_json('letter_dict.json')
app.config['letter_map'] = alphabet_dict

device = ( #selects device
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )

config_dict = read_json('config.json')

conv1 = config_dict['conv1']
ckernel1 = config_dict['ckernel1']
MPkernel1 = config_dict['MPkernel1']
conv2 = config_dict['conv2']
ckernel2 = config_dict['ckernel2']
MPkernel2 = config_dict['MPkernel2']
conv3 = config_dict['conv3']
ckernel3 = config_dict['ckernel3']
MPkernel3 = config_dict['MPkernel3']
fc1 = config_dict['fc1']

model = CNN(
    input_channels=1,
    conv1=conv1,
    ckernel1=ckernel1,
    MPkernel1=MPkernel1,
    conv2=conv2,
    ckernel2=ckernel2,
    MPkernel2=MPkernel2,
    conv3=conv3,
    ckernel3=ckernel3,
    MPkernel3=MPkernel3,
    fc1=fc1,
    out_dim=26,
    input_HW=28,
    device=device
)

model_path = 'models/2024-09-30 18_13_15.758843'
state_dict = torch.load(model_path,map_location=torch.device(device),weights_only=True)
model.load_state_dict(state_dict=state_dict)
app.config['model'] = model
app.config['device'] = device

@app.route("/")
def home():
    file_name = session.get('uploaded_filename',None)
    resized_img = session.get('resized_img_path', None)
    letter = session.get('letter',None)
    sorted_top_3 = session.get('sorted_top_3',None)
    print(file_name)
    return flask.render_template('index.html',file_name=file_name, resized_img = resized_img, letter = letter, sorted_top_3=sorted_top_3)

@app.route("/submit", methods=['POST'])
def submit():
    file = request.files['img_upload']
    if file.filename == '': 
        flash('No file was uploaded')
        return redirect(url_for('home'))
    file_path = os.path.join(app.config['UPLOAD_FOLDER'],'uploaded_file')
    file.save(file_path) #save image
    file_name = file.filename
    #file name for user:
    session['uploaded_filename'] = secure_filename(file_name) #store file name - display to user
    flash(f"File '{file_name}' uploaded successfully!", "success") #message

    #save inverted image:
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_file')
    image = PIL.Image.open(image_path)
    image = image.convert('L')
    inverted_image = PIL.ImageOps.invert(image) #invert colors, white background -> black background
    contrasted_inv_img:PIL.Image.Image = tvF.adjust_contrast(inverted_image,2)
    threshold_value = 100
    thresholded_img = contrasted_inv_img.point(lambda p: p if p > threshold_value else 0)
    #inverted_image = image
    
    resize_transform = v2.Resize([28,28])
    resized_img:PIL.Image.Image = resize_transform(thresholded_img)
    resized_img_path = 'static/process_imgs/resized_img.jpg'
    session['resized_img_path'] = resized_img_path
    resized_img.save(resized_img_path)

    return redirect(url_for('home'))

@app.route('/predict',methods=['POST'])
def predict():
    resized_inv_image = PIL.Image.open(session['resized_img_path'])

    image_transform = image_transform = v2.Compose([
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # to tensor
    v2.Normalize((0.1736,),(0.3248,))])

    transformed_img:torch.Tensor = image_transform(resized_inv_image)
    transformed_img = transformed_img.unsqueeze(0)

    model:torch.nn.Module = app.config['model']
    model.eval()
    device = app.config['device']
    transformed_img = transformed_img.to(device=device)
    preds = model(transformed_img)
    probabilities = torch.softmax(preds,1).detach().cpu().numpy()
    probabilities = probabilities.squeeze(0)
    most_probable = int(np.argmax(probabilities))
    top_3_indices = np.argsort(probabilities)[-3:]
    top_3_probs = [round(float(probabilities[i])*100,2) for i in top_3_indices]

    top_3_alpha = [app.config['letter_map'][str(idx)] for idx in top_3_indices]
    
    most_probable_alpha = app.config['letter_map'][str(most_probable)]
    session['letter'] = most_probable_alpha
    top_3_joined = list(zip(top_3_alpha,top_3_probs)) #already sorted
    sorted_top_3 = sorted(top_3_joined, key=lambda x: x[1], reverse=True) #sorts with highest prob first

    session['sorted_top_3'] = sorted_top_3
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(port=8000,debug=True)