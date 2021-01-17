import numpy as np
import os

#image_name = 'COCO_val2014_000000262148'

#temp = {'image_name': 'COCO_val2014_000000262148', 'image_path': '/scratch1/nvishwa/datasets/VQA/Resized_Images/val2014/COCO_val2014_000000262148.jpg', 'questions': []}

vqa_mm_corruption_attack = []

vqa = np.load(os.path.join('/scratch1/nvishwa/datasets/VQA', 'valid.npy'), allow_pickle = True)

# load image names
image_names = set()
for i in vqa:
  image_names.add(i['image_name'])  

for image_name in image_names:
  temp = {'image_name': image_name, 'questions': []} # keeps adding image_path, but its oK
  for i in vqa:
    if i['image_name'] == image_name:
      temp_i = dict(i)
      del temp_i['image_name']
      del temp_i['image_path']
      temp['questions'].append(temp_i)
      temp['image_path'] = i['image_path']

  vqa_mm_corruption_attack.append(temp)

np.save('/scratch1/nvishwa/datasets/VQA/valid_mm_corruption_attack_full.npy', vqa_mm_corruption_attack)
