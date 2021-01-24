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

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        embedding_s = self.embed(captions[:,:-1])
        embedding_s = torch.cat((features.unsqueeze(dim=1), embedding_s), dim=1)
        lstm_out_1, _ = self.lstm(embedding_s)
        linear_layer_s = self.linear(lstm_out_1)
        return linear_layer_s
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sentence_prediction = []
        
        for index in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            linear_layer = self.linear(lstm_out.squeeze(1))
            max_probability = linear_layer.max(1)[1]
            sentence_prediction.append(max_probability.item())
            inputs = self.embed(max_probability).unsqueeze(dim=1)
            
        return sentence_prediction

    
    
    
            
            