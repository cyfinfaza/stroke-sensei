import cv2 # type: ignore
import numpy as np # type: ignore
import random
import os
import matplotlib.pyplot as plt # type: ignore


def load_kanji_image(image_path, target_size=(28, 28)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, target_size)  
    return img

def rotate_image(img, max_angle=15):
    rows, cols = img.shape
    angle = random.uniform(-max_angle, max_angle)
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows))

def flip_image(img):
    return cv2.flip(img, 1)  # Flip horizontally

def adjust_brightness(img, factor=1.2):
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def add_noise(img):
    noise = np.random.normal(0, 15, img.shape)  # Gaussian noise
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_img

def adjust_contrast(img, factor=1.5):
    mean = np.mean(img)
    return np.clip((1 + factor) * (img - mean) + mean, 0, 255).astype(np.uint8)

def random_shift(img, max_shift=5):
    rows, cols = img.shape
    M = np.float32([[1, 0, random.randint(-max_shift, max_shift)], 
                    [0, 1, random.randint(-max_shift, max_shift)]])
    return cv2.warpAffine(img, M, (cols, rows))

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def manipulate_kanji_image(image_path, output_dir, num_images=20):
    img = load_kanji_image(image_path)  


    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        transformation = random.choice(['rotate', 'flip', 'brightness', 'noise', 'contrast', 'shift'])
        
        if transformation == 'rotate':
            transformed_img = rotate_image(img)
        elif transformation == 'brightness':
            transformed_img = adjust_brightness(img, factor=random.uniform(0.8, 1.2))
        elif transformation == 'noise':
            transformed_img = add_noise(img)
        elif transformation == 'contrast':
            transformed_img = adjust_contrast(img, factor=random.uniform(1.0, 2.0))
        elif transformation == 'shift':
            transformed_img = random_shift(img)

        cv2.imwrite(os.path.join(output_dir, f"kanji_2_{i + 100}.png"), transformed_img)

image_path = '../kanji_images/kanji_2/image_2.png'  
output_dir = '../kanji_images/kanji_2/'  
manipulate_kanji_image(image_path, output_dir, num_images=100)

print(f"Augmented images saved to: {output_dir}")