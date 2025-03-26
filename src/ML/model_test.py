import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np # type: ignore


model = tf.keras.models.load_model('kanji_model.h5')

#preprocess image
img_path = '../ML/input_images/kanji_drawing_1.png'  
img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img) / 255.0  
img_array = img_array.reshape((1, 28, 28, 1))


prediction = model.predict(img_array)


predicted_class_index = np.argmax(prediction) 


class_labels = ['体', '日']  
predicted_kanji = class_labels[predicted_class_index]

print(f"Predicted kanji: {predicted_kanji}")
