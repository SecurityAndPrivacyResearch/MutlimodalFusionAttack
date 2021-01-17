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

def get_adv(args):
  # prepare anchor text
  vqa = np.load(args.input_dir+'/' + 'valid.npy', allow_pickle = True)
  qst_vocab = text_helper.VocabDict(args.input_dir + '/vocab_questions.txt')
  max_qst_length = args.max_qst_length
  adv_text = ['<unk>']
  qst2idc = np.array([qst_vocab.word2idx('<pad>')] * args.max_qst_length)  # padded with '<pad>' in 'ans_vocab'
  qst2idc[:len(adv_text)] = [qst_vocab.word2idx(w) for w in adv_text]
  t_a = torch.tensor(qst2idc).unsqueeze(0).cuda()

  # prepare anchor image
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  image_path = '/scratch1/nvishwa/datasets/VQA/Resized_Images/val2014/COCO_val2014_000000262323.jpg'
  image = Image.open(image_path).convert('RGB')
  x_a = transform(image)
  x_a = x_a.unsqueeze(0).cuda()
  return x_a, t_a

def multimodal_objective(x, x_prime, z_prime, z_a, z_t_i, cos_loss1, cos_loss2, m = 0.5, lambda1 = 1.8, lambda2 = 1.5):

  loss1 = torch.max( torch.tensor(0.).cuda(), m + lambda1 * cos_loss1(z_prime, z_a, torch.ones(1).cuda()) - lambda2 * cos_loss2(z_prime, z_t_i, torch.ones(1).cuda()) )

  return loss1

# if 80% of results are successful, then we consider this a success
def Phi(results):
  trues = 0.
  falses = 0.
  for r in results:
    if r == True:
      trues += 1
    else:
      falses += 1
  if (trues / len(results)) * 100 >= 80:
    return True
  else:
    return False

def attack(args, model, x, questions, labels, question_ids, image_name):

  # prepare adversarial example variable
  x_prime = x.detach().clone().requires_grad_(True).cuda()

  # prepare optimizer
  optimizer = optim.Adam([x_prime], lr = 1e-3)

  # prepare anchor
  x_a, t_a = get_adv(args)
  a_out, z_a, _, _ = model(x_a, t_a)
  z_a.detach()

  # cosine loss
  cos_loss1 = nn.CosineEmbeddingLoss()
  cos_loss2 = nn.CosineEmbeddingLoss()

  final_success = False

  for i in range(500):

    is_success = True
    total_loss = 0.
  
    optimizer.zero_grad()

    for question_id, question, label in zip(question_ids, questions, labels):
    
      question = question.cuda()
      label = label.cuda()

      # right/orig prediction
      xti_out, z_i, _, _ = model(x, question)
      z_i.detach()
      _, pred_exp_orig = torch.max(xti_out, 1)
      print('Label:', label, 'Prediction:', pred_exp_orig)

      x_prime.retain_grad()
      out, z_prime, _, _ = model(x_prime, question)
      _, pred_exp_adv = torch.max(out, 1)

      loss = multimodal_objective(x, x_prime, z_prime, z_a, z_i, cos_loss1, cos_loss2)

      total_loss += loss

    total_loss += F.mse_loss(x_prime, x)
    total_loss.backward(retain_graph = True)

    optimizer.step()

    # check if decision boundary is crossed. TODO: ENclose in no grad
    results = []
    for question_id, question, label in zip(question_ids, questions, labels):
      question = question.cuda()
      label = label.cuda()
      out, _, _, _ = model(x_prime, question)
      _, pred_exp_adv_check = torch.max(out, 1)
    
      print('It:', i, 'Pred of adv:', pred_exp_adv_check, 'Loss: %.4f' % (total_loss) )
      print()

      if pred_exp_adv_check != label:
        #is_success = is_success and True
        results.append(True)
      else:
        #is_success = is_success and False
        results.append(False)

    #if is_success == True and i >= 50:
    if Phi(results) == True and i >= 50:
      final_success = True
      break

  pert = x_prime.grad.squeeze().cpu().detach().numpy().transpose(1,2,0) * 255
  img = x_prime.squeeze().cpu().detach().numpy().transpose(1,2,0) * 255

  torch.save(x_prime, os.path.join('adversarial_samples', image_name[0] + '_' + str(final_success) + '.pt')) # I save the adversarial directly, since saving as image and then reloading removes adversarial effect.

  imageio.imwrite(os.path.join('adversarial_samples', image_name[0] + '_' + str(final_success) + '_adv.png'), img)
  imageio.imwrite(os.path.join('adversarial_samples', image_name[0] + '_' + str(final_success) + '_pert.png'), pert)
  
  #imageio.imwrite(str(question_id.item()) + str(is_success) + '_adv.png', img)
  #imageio.imwrite(str(question_id.item()) + str(is_success) + '_pert.png', pert)
  
  print('-' * 20)








