import os
import random
import shutil

dataset_dir = "dataset" 
train_dir = "train"
test_dir = "test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for person in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person)
    
    if not os.path.isdir(person_path):
        continue  
    
  
    images = [img for img in os.listdir(person_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random.shuffle(images)

    train_images = images[:100]
    test_images = images[100:120]
    remaining_images = images[120:] 

    train_person_dir = os.path.join(train_dir, person)
    test_person_dir = os.path.join(test_dir, person)
    os.makedirs(train_person_dir, exist_ok=True)
    os.makedirs(test_person_dir, exist_ok=True)

    for img in train_images:
        shutil.move(os.path.join(person_path, img), os.path.join(train_person_dir, img))
    
    for img in test_images:
        shutil.move(os.path.join(person_path, img), os.path.join(test_person_dir, img))

    for img in remaining_images:
        os.remove(os.path.join(person_path, img))

print("Data split into train and test folders successfully!")