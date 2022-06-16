from textblob import TextBlob
import random
import os
from transformers import BertTokenizer, BertModel
import torch
import numpy 
import os



tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained("bert-large-uncased")

model.eval().cuda()

for cls in os.listdir("../Bird/text/"):
#     os.makedirs("../data/fg2/Flowers102/text_feature/" + cls)
    for item in os.listdir("../Bird/text/" + cls):
        print("../Bird/text/" + cls + "/" + item)
        f = open("../Bird/text/" + cls + "/" + item, "r")
        l = []
        for line in f.readlines():
            blob = TextBlob(line)
            nouns = blob.noun_phrases
            l = l + list(nouns)
#         print(l)
        text = " ".join(list(set(l)))
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)['last_hidden_state'].squeeze(0)
        output = torch.mean(output, 0).cuda().detach().numpy()
#         output = output.cpu().detach().numpy()
#         print(output.shape)
        numpy.save("../Bird/text_feature/" + cls + "/" + item.split('.')[0] +".npy", output)
