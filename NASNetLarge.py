from keras.applications.nasnet import NASNetLarge,preprocess_input,decode_predictions
from keras.preprocessing import image
from keras.models import Model
import glob
import cv2
import numpy as np



model = NASNetLarge( weights = 'imagenet', classes = 1000)

for image_path in glob.iglob('images/*'):

    frame = cv2.imread(image_path)
    frame = cv2.resize(frame , (400,300) )

    img = image.load_img(image_path,target_size =(331,331) )
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis = 0)
    x = preprocess_input(x)

    pred = model.predict(x)

    label = decode_predictions(pred,top = 1)[0][0][1]

    if decode_predictions(pred,top = 1)[0][0][2] > 0.2:

        cv2.putText(frame,label,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        cv2.imshow("frame",frame)
        key = cv2.waitKey(1000000) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()

