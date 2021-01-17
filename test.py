import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#from data_loader import get_loader
from mm_corruption_attack_data_loader import get_loader
from models import VqaModel
from utils import text_helper
import torch.nn.functional as F
import cv2
import imageio
from advs import *
from mm_corruption_attack import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    data_loader = get_loader(
        input_dir=args.input_dir,
        input_vqa_train='train.npy',
        #input_vqa_valid='valid.npy',
        input_vqa_valid='valid_mm_corruption_attack.npy',
        max_qst_length=args.max_qst_length,
        max_num_ans=args.max_num_ans,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx
 
    # Load model checkpoint
    checkpoint = torch.load('/scratch1/nvishwa/datasets/VQA/models/model-epoch-20.ckpt')
    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size).to(device)

    model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.CrossEntropyLoss()

    params = list(model.img_encoder.fc.parameters()) \
        + list(model.qst_encoder.parameters()) \
        + list(model.fc1.parameters()) \
        + list(model.fc2.parameters())

    batch_step_size = len(data_loader['valid'].dataset) / args.batch_size

    #model.eval()
    model.train()

    max_samples = 500

    for batch_idx, batch_sample in enumerate(data_loader['valid']):

        image = batch_sample['image'].to(device)
        question_id = batch_sample['question_id']
        question = batch_sample['question'].to(device)
        label = batch_sample['answer_label'].to(device)
        multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

        #print(image_path)

        with torch.set_grad_enabled(True):

            #attack(args, model, image, question, label, question_id) 
            attack(args, model, data_loader['valid']) 

        if batch_idx > max_samples:
          break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='/scratch1/nvishwa/datasets/VQA',
                        help='input directory for visual question answering.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='/scratch1/nvishwa/datasets/VQA/models',
                        help='directory for saved models.')

    parser.add_argument('--max_qst_length', type=int, default=30,
                        help='maximum length of question. \
                              the length in the VQA dataset = 26.')

    parser.add_argument('--max_num_ans', type=int, default=10,
                        help='maximum number of answers.')

    parser.add_argument('--embed_size', type=int, default=1024,
                        help='embedding size of feature vector \
                              for both image and question.')

    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='embedding size of word \
                              used for the input in the LSTM.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of the RNN(LSTM).')

    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size in the LSTM.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=10,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=1, # default batch size is 256
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=1, # default was 8
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    args = parser.parse_args()

    main(args)
