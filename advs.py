import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils import text_helper
import torch.nn.functional as F
import imageio
import os

def get_adv_text(args):
  vqa = np.load(args.input_dir+'/' + 'valid.npy', allow_pickle = True)
  qst_vocab = text_helper.VocabDict(args.input_dir + '/vocab_questions.txt')
  max_qst_length = args.max_qst_length
  #adv_text = ['where', 'is', 'he', 'sitting', '?']
  #adv_text = ['what', 'is', 'he', 'doing', '?']
  #adv_text = ['are', 'there', 'chopsticks', 'in', 'the', 'picture', '?']
  #adv_text = ['is', 'there', 'soup', '?']
  #adv_text = ['table']
  adv_text = ['is', 'he', 'looking', '?']
  #adv_text = ['what', 'is', 'above', 'him', '?']
  qst2idc = np.array([qst_vocab.word2idx('<pad>')] * args.max_qst_length)  # padded with '<pad>' in 'ans_vocab'
  qst2idc[:len(adv_text)] = [qst_vocab.word2idx(w) for w in adv_text]
  return torch.tensor(qst2idc).unsqueeze(0).cuda()

def multimodal_objective_untargeted(x, x_prime, z_prime, z_p, output, cos_loss, ignore_zp):
  # get max current logit
  current_max = torch.argmax(output)
  #import IPython; IPython.embed(); exit(1)
  loss1 = F.cross_entropy(output, current_max.unsqueeze(0))
  if ignore_zp == False:
    loss2 = cos_loss(z_prime, z_p, torch.ones(1).cuda()) # minimize this loss
  else:
    loss2 = 0.
  loss3 = F.mse_loss(x_prime, x)

  # TODO: Need to parameterize these losses
  return loss1 + loss2 + loss3

def multimodal_objective(x, x_prime, z_prime, z_p, z_t, output, target, cos_loss1, cos_loss2, ignore_zp):
  loss1 = F.cross_entropy(output, target)
  loss2 = cos_loss1(z_prime, z_t, torch.ones(1).cuda())
  if ignore_zp == False:
    loss3 = cos_loss2(z_prime, z_p, -1. * torch.ones(1).cuda())
  else:
    loss3 = 0.
  loss4 = F.mse_loss(x_prime, x)

  # TODO: Need to parameterize these losses
  return loss1 + 1.5 * loss2 + 0.4 * loss3 + 0.4 * loss4

def ranking_loss(z_prime, zt, z_orig, m = 0.5):
  #import IPython; IPython.embed(); exit(1)
  return torch.max(torch.tensor(0.).cuda(), m - torch.norm(z_prime - z_orig, 1) + torch.norm(z_prime - zt, 1))
  #return m + torch.norm(z_prime - zt, 1) - torch.norm(z_prime - z_orig, 1)

def atanh(x, eps=1e-6):
  x = x * (1 - eps)
  return 0.5 * torch.log((1.0 + x) / (1.0 - x))

def to_tanh_space(x, box = (-1., 1.)):
  _box_mul = (box[1] - box[0]) * 0.5
  _box_plus = (box[1] + box[0]) * 0.5
  return atanh((x - _box_plus) / _box_mul)

def multimodal_attack(args, model, x, question, label, question_id, targeted = True):

  if targeted == True:
    save_dir = 'adversarial_samples'
  else:
    save_dir = 'adversarial_samples_untargeted'

  print('Orig label:', label)
  out_orig, z_orig, _, _ = model(x, question)
  z_orig.detach()
  _, pred_exp_orig = torch.max(out_orig, 1)

  if label.item() != pred_exp_orig:
    print('Model did not predict correct label, skipping ...')
    return

  # prepare adversarial example variable
  x_prime = x.detach().clone().requires_grad_(True).cuda()

  # prepare optimizer
  optimizer = optim.Adam([x_prime], lr = 1e-3)

  # prepare target
  adv_question = get_adv_text(args)
  adv_out, z_t, _, _ = model(x, adv_question)
  z_t.detach()
  _, adv_label = torch.max(adv_out, 1)
  #adv_label = torch.tensor([5]).cuda()
  print('Target label:', adv_label)

  # cosine loss
  cos_loss1 = nn.CosineEmbeddingLoss()
  cos_loss2 = nn.CosineEmbeddingLoss()

  z_p = None

  is_success = False

  for i in range(1500):
    optimizer.zero_grad()

    x_prime.retain_grad()
    out, z_prime, _, _ = model(x_prime, question)
    _, pred_exp_adv = torch.max(out, 1)

    if targeted == True:
      if i == 0 or pred_exp_adv == adv_label:
        loss = multimodal_objective(x, x_prime, z_prime, z_p, z_t, out, adv_label, cos_loss1, cos_loss2, True)
      else:
        loss = multimodal_objective(x, x_prime, z_prime, z_p, z_t, out, adv_label, cos_loss1, cos_loss2, False)
    else:
      if i == 0:
        loss = multimodal_objective_untargeted(x, x_prime, z_prime, z_p, out, cos_loss1, True)
      else:
        loss = multimodal_objective_untargeted(x, x_prime, z_prime, z_p, out, cos_loss1, False)

    loss.backward(retain_graph = True)

    optimizer.step()
    
    # preprare z_p for next iteration
    z_p = z_prime.detach().clone().cuda()

    print('It:', i, 'Pred of adv:', pred_exp_adv, 'Loss: %.4f' % (loss) )
    if targeted == True:
      if pred_exp_adv == adv_label:
        is_success = True
        break
    else:
      if pred_exp_adv != label:
        is_success = True
        break

  pert = x_prime.grad.squeeze().cpu().detach().numpy().transpose(1,2,0) * 255
  img = x_prime.squeeze().cpu().detach().numpy().transpose(1,2,0) * 255

  #name = image_path[0].split('/')[-1].split('.')[0] + str(is_success)

  imageio.imwrite(os.path.join(save_dir, str(question_id.item()) + str(is_success) + '_adv.png'), img)
  imageio.imwrite(os.path.join(save_dir, str(question_id.item()) + str(is_success) + '_pert.png'), pert)
  #import IPython; IPython.embed(); exit(1)
  print('-' * 20)








