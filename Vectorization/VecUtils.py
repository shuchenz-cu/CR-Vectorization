from io import BytesIO

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable


class ImgVectorize:
    def __init__(self, url):
        self.url = url
        self.model = models.resnet34(pretrained='imagenet')
        #Resize the image to 224x224 px
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        # Use the model object to select the desired layer
        self.layer = self.model._modules.get('avgpool')


    def extract_feature_vector(self, img):
        # 2. Create a PyTorch Variable with the transformed image
        #Unsqueeze actually converts to a tensor by adding the number of images as another dimension.
        t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))

        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(1, 512, 1, 1)

        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        # 5. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)

        # 6. Run the model on our transformed image
        self.model(t_img)

        # 7. Detach our copy function from the layer
        h.remove()

        # 8. Return the feature vector
        return my_embedding.squeeze().numpy()

    def get_vec(self):
        response = requests.get(self.url, stream=True)
        if response.status_code == 403:
            return [[0]]
        else:
            try:
                img = Image.open(response.raw)
                return self.extract_feature_vector(img).reshape(1, -1)
            except:
                return "invalid image link"




class TextVectorize:
    import torch
    from transformers import BertTokenizer, BertModel

    # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
    import logging
    #logging.basicConfig(level=logging.INFO)
    #logging.set_verbosity_error()

    import matplotlib.pyplot as plt
    # % matplotlib inline

    # Load pre-trained model tokenizer (vocabulary)
    def __init__(self, sentence):
        from transformers import BertTokenizer, BertModel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.str = sentence

    def get_vec(self):
        from transformers import BertTokenizer, BertModel
        #1.Tokenize the sequence:
        tokens=self.tokenizer.tokenize(self.str)
        #print(len(tokens)) #new
        #print(tokens)

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        #print(tokens) #new
        #print(" Tokens are \n {} ".format(tokens))
        

        T=150 #T=15 #untuk mengoreksi jumlah kata kalo terlalu sedikit
        padded_tokens=tokens +['[PAD]' for _ in range(T-len(tokens))] #ditambah remark PAD untuk token2 akhir
        #print("Padded tokens are \n {} ".format(padded_tokens))
        
        #NEW ADDING THE [SEP]
        padded_tokens=['[SEP]' if token == '/' else token for token in padded_tokens] #new
        #print("Add SEP are \n {} ".format(padded_tokens))
        
        #attn_mask=[ 1 if token != '[PAD]' else 0 for token in padded_tokens] #input mask,  ditandanin yg ada word dengan 1, emark PAD dikasih 0
        attn_mask=[ 1 if token != '[PAD]' and token !='[SEP]' else 0 for token in padded_tokens] #new
        #print("Attention Mask are \n {} ".format(attn_mask))

        #seg_ids=[0 for _ in range(len(padded_tokens))] # semua token dikasin nilai 0, untuk separator?
        seg_ids=[1 if token =='[SEP]' else 0 for token in padded_tokens] # new
        #print("Segment Tokens are \n {}".format(seg_ids))

        sent_ids=self.tokenizer.convert_tokens_to_ids(padded_tokens) #tiap token dikasih unique id, CLS 101, SEP 102
        #print("senetence idexes \n {} ".format(sent_ids))
        
        # DI bawah ini intinya dijaddin formatnya torch tensor
        token_ids = torch.tensor(sent_ids).unsqueeze(0)
        #print("token idexes \n {} ".format(token_ids)) #new
        attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
        #print("attn mask \n {} ".format(attn_mask)) #new
        seg_ids   = torch.tensor(seg_ids).unsqueeze(0)
        #print("seg ids \n {} ".format(seg_ids)) #new
        
        

        model = BertModel.from_pretrained('bert-base-uncased',
                                        output_hidden_states = True, # Whether the model returns all hidden-states.
                                        )

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()

        with torch.no_grad():

            outputs = model(token_ids, seg_ids)
            hidden_states = outputs[2]

        # hidden_reps, cls_head = BertModel(token_ids, attention_mask = attn_mask,token_type_ids = seg_ids)
        # print(type(hidden_reps))
        # print(hidden_reps.shape ) #hidden states of each token in inout sequence 
        # print(cls_head.shape ) #hidden states of each [cls]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding
