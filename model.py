import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        torch.nn.init.xavier_uniform_(self.embed.weight)   # initialize weights using uniform random distribution
        
        self.batchNorm = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batchNorm(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #create an embedding layer to turn caption words into an input vector matching the flattened image feature vector
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        torch.nn.init.xavier_uniform_(self.word_embeddings.weight)   # initialize weights using uniform random distribution

        dropout = 0.0
        if num_layers > 1:
            dropout = 0.2
        
        # define the LSTM that converts the input (embed_size) to hidden state (hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # define the linear layer that converts the hidden state to the output (word likelihood for each word in caption)
        self.lstm2Caption = nn.Linear(hidden_size, vocab_size)
        torch.nn.init.xavier_uniform_(self.lstm2Caption.weight)   # initialize weights using uniform random distribution
       
    def forward(self, features, captions):
        batch_size = captions.shape[0]
        caption_length = captions.shape[1]
        
        # removed end tokens from all captions
        captions = captions[:, :-1]
        
        embeds = self.word_embeddings(captions)   # create embedded word vectors for each word in caption
        #print("embeds.shape: ", embeds.shape)
        
        # merge features and captions along axis 1 (axis 1 is along caption length). features will be the first token in each caption
        merged = torch.cat((features.unsqueeze(1), embeds), 1)
        #print("merged.shape: ", merged.shape)
        
        # initialize hidden_state using uniform random distribution 
        hidden_state_tuple = (torch.randn(self.num_layers, batch_size, self.hidden_size).to(features.device),
                              torch.randn(self.num_layers, batch_size, self.hidden_size).to(features.device))
        #print("h0, c0 shapes: ", hidden_state_tuple[0].shape, hidden_state_tuple[1].shape)
        
        lstm_out, _ = self.lstm(merged, hidden_state_tuple)
        
        predicted_captions = self.lstm2Caption(lstm_out)
        
        return predicted_captions

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_word_indices = []
        
        # necessary to initialize states tuple to avoid repetitive phrases - use uniform random distribution
        states = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device), 
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        #print("h0, c0 shapes: ", states[0].shape, states[1].shape)
        
        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            predicted_word_likelihoods = self.lstm2Caption(lstm_out)   # output is 1 x 1 x vocab

            #get the word index with the maximum likelihood and append to our list
            predicted_word_index = torch.max(predicted_word_likelihoods, 2)[1]   # we want the argmax (index). Not the likelihood
            #print(predicted_word_index.shape)
            
            predicted_word_indices.append(predicted_word_index.item())
            
            if predicted_word_index == 1:   # end tag
                break
            
            # create embedded word vector for captions (should be 1 x 1 x 512)
            inputs = self.word_embeddings(predicted_word_index)   # input is 1 x 1
            #print(inputs.shape)          
                    
        return predicted_word_indices