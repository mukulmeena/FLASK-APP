#This is a sample model to demonstrate how a Machine Learning model 
#can be implemented in Production as a REST API and how it can be consumed.


#Import libraries and packages
import pickle
import numpy as np
from flask import Flask, jsonify, request, render_template



#Create Flask object to run and specify the template folder to render your html file. 
app = Flask(__name__, template_folder="Template")



#Create endpoints for your application
@app.route('/') # Homepage 
def hello():
     return render_template('index.html') # Render the Homepage




#It will predict the Iris species and return the result to webpage
@app.route('/predict') 
def predict() -> int:
    ''' For rendering results on HTML GUI
    '''
    
    #Take Values from the form through URL
    sep_Len = request.args.get('slen')
    sep_Wid = request.args.get('swid')
    pet_Len = request.args.get('plen')
    pet_Wid = request.args.get('pwid')

    final_features = np.array([sep_Len, sep_Wid, pet_Len, pet_Wid]).reshape(1, 4)
    prediction = LR_model.predict(final_features)
    output = "Predicted Iris Class: " + str(prediction[0])

    
    #To render the image of predicted Iris Class
    if str(prediction[0]) == 'Iris-setosa':
        image_name ='Iris_setosa.jpg' 
    elif str(prediction[0]) == 'Iris-versicolor':
        image_name ='Iris_versiclor.jpg'
    else:
        image_name ='Iris_verginica.jpg'
    
    return render_template('output.html', image= image_name, variety= output) #rendering the predicted result 




##This method is created for taking post request from the 
#browser in the form of JSON file
@app.route('/api/predict', methods = ["POST"]) 
def predict_api():
    dict = request.get_json()
    l=[]
    for i in dict:
        l.append(dict[i])
    
    final_features = np.array(l).reshape(1,4)
    output = LR_model.predict(final_features)
    ret_json = {
        "The Predicted Iris class is: " : output[0]
    }

    return jsonify(ret_json)


##Load the pre-trained and saved model
#Note: The model will be loaded only once at the start of the server
def model():
    
    global LR_model
    LR_model = pickle.load(open('model/LRModel.pckl', 'rb'))



if __name__=='__main__':
    
    #call function that loads model
    model()
    
    #Run server
    app.run()