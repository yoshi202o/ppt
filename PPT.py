import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

video = cv2.VideoCapture(0)

while(True):
    check, frame = video.read()
    
    #Redimensionar la imagen
    img = cv2.resize(frame,(224,224))
    
    #Convertir la imagen a matriz Numpy y la vuelvo a hacer grande
    prueba1= np.array(img, dtype=np.float32)
    prueba1= np.expand_dims(prueba1, axis=0)

    #Regresar a imagen para poder verla nuevanente
    img_normal = prueba1/255

    #Resultados de la predicción
    p = model.predict(img_normal)
    rock = int(p[0][0]*100)
    paper = int(p[0][1]*100) 
    scissor = int(p[0][2]*100)
    
    print ("Predicion : ", p )
    
    cv2.imshow("Resultado", frame)
    
    key = cv2.waitKey(1)
    
    if key == 32:
        print("CErando")
        break
    
video.release()    