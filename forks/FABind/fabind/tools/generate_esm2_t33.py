import torch
import esm
from tqdm import tqdm
import lmdb
import os
import pickle
import sys

data_path = os.path.join(sys.argv[1], 'dataset/processed')

device = "cuda" if torch.cuda.is_available() else "cpu"

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                'N': 2, 'Y': 18, 'M': 12}

num_to_letter = {v:k for k, v in letter_to_num.items()}

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()

protein_db = lmdb.open(os.path.join(data_path, 'protein_1d_3d.lmdb'), readonly=True)
protein_esm2_db = lmdb.open(os.path.join(data_path, 'esm2_t33_650M_UR50D.lmdb'), map_size=1024 ** 4)

with protein_db.begin(write=False) as txn:
    count = 0
    for _ in txn.cursor():
        count += 1
print(count)

with protein_db.begin(write=False) as txn:
    with protein_esm2_db.begin(write=True) as txn_esm2:
        cursor = txn.cursor()
        for key, value in tqdm(cursor, total=count):
            pdb_id = key.decode()
            seq_in_id = pickle.loads(value)[1].tolist()
            seq_in_str = ''.join([num_to_letter[a] for a in seq_in_id])

            data = [
                ("protein1", seq_in_str),
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
            token_representations = results["representations"][33][0][1: -1]
            assert token_representations.shape[0] == len(seq_in_str)

            txn_esm2.put(pdb_id.encode(), pickle.dumps(token_representations.cpu()))

protein_db.close()
protein_esm2_db.close()