import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils import text_helper
import torch.nn.functional as F
import torchvision.transforms as transforms
import imageio
from PIL import Image
import os

def get_inference_sample(args):
  # prepare sample text
  vqa = np.load(args.input_dir+'/' + 'valid.npy', allow_pickle = True)
  qst_vocab = text_helper.VocabDict(args.input_dir + '/vocab_questions.txt')
  max_qst_length = args.max_qst_length
  #adv_text = ['what', 'animal', 'is', 'in', 'the', 'image', '?']
  #adv_text = ['is', 'there', 'a', 'skateboard', 'in', 'the', 'image', '?']
  #adv_text = ['what', 'is', 'below', 'the', 'human', '?']
  #adv_text = ['is', 'there', 'a', 'giraffe', '?']
  adv_text = ['is', 'there', 'a', 'person', '?']
  #adv_text = ['what', 'is', 'the', 'person', 'doing', '?']
  qst2idc = np.array([qst_vocab.word2idx('<pad>')] * args.max_qst_length)  # padded with '<pad>' in 'ans_vocab'
  qst2idc[:len(adv_text)] = [qst_vocab.word2idx(w) for w in adv_text]
  t_a = torch.tensor(qst2idc).unsqueeze(0).cuda()

  # prepare sample image, use adversarial image
  #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  #image_path = '/home/nvishwa/pytorch_stuff/MULTIMODAL_ADV/basic_vqa/262148002True_adv.png'
  #image = Image.open(image_path).convert('RGB')
  #x_a = transform(image)
  #x_a = x_a.unsqueeze(0).cuda()

  x_a = torch.load('x_prime.pt')

  return x_a, t_a

def infer(args, model):

  # prepare anchor
  x, t = get_inference_sample(args)
  out, _, _, _ = model(x, t)
  _, pred = torch.max(out, 1)
  print('Output:', pred)
  #import IPython; IPython.embed(); exit(1)









