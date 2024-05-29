import os
import gzip
import pickle
import requests
import re
import torch
from transformers import BertTokenizer, BertModel
from time import time
from tqdm import tqdm

def generate_protbert_features():
    t0 = time()
    modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
    configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
    vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'

    downloadFolderPath = ''

    modelFolderPath = downloadFolderPath

    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')

    configFilePath = os.path.join(modelFolderPath, 'config.json')

    vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')

    if not os.path.exists(modelFolderPath):
        os.makedirs(modelFolderPath)

    def download_file(url, filename):
        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                           total=int(response.headers.get('content-length', 0)),
                           desc=filename) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)

    if not os.path.exists(modelFilePath):
        download_file(modelUrl, modelFilePath)

    if not os.path.exists(configFilePath):
        download_file(configUrl, configFilePath)

    if not os.path.exists(vocabFilePath):
        download_file(vocabUrl, vocabFilePath)

    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False)
    model = BertModel.from_pretrained(modelFolderPath)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model = model.eval()

    sequences = []

    with open('', 'r') as f:
        protein_list = f.readlines()
        for protein in protein_list:
            seq = open('data/Novel/NBP/{}.fasta'.format(protein.strip()), 'r').readlines()
            sequences += [seq[1].strip()]

    sequences_Example = [' '.join(list(seq)) for seq in sequences]
    sequences_Example = [re.sub(r"[-UZOB]", "X", sequence) for sequence in sequences_Example]

    all_protein_features = []

    for i, seq in enumerate(sequences_Example):
        ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # 获取最后一层隐藏状态

        last_hidden_state = last_hidden_state[0, -1, :]  # 仅选择第一个序列的最后一个隐藏状态
        last_hidden_state = last_hidden_state.cpu().numpy()
        all_protein_features.append(last_hidden_state)

    # 将所有特征向量保存到文件
    pickle.dump({'ProtBert_features': all_protein_features},
                gzip.open('ProtBert_features_NBP.pkl.gz', 'wb'))

    print('Total time spent for ProtBERT:', time() - t0)

if __name__ == '__main__':
    generate_protbert_features()

