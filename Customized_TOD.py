import torch.nn as nn
from transformers.modeling_outputs import MultipleChoiceModelOutput

class TOD(nn.Module):
    """
    Customized TOD Bert 
    """
    def __init__(self, root_model, num_classes):
        super(TOD, self).__init__()
        self.root_model = root_model
        self.classifier = nn.Linear(root_model.config.hidden_size, 1, bias=True)
        # self.softmax = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        dim0, dim1 = input_ids.shape[0], input_ids.shape[1]
        input_ids = input_ids.view(dim0*dim1, -1)
        attention_mask = attention_mask.view(dim0*dim1, -1)
        token_type_ids = token_type_ids.view(dim0*dim1, -1)
        x = self.root_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.classifier(x[0]) #pass last_hidden_state of [cls] through dense layer
        x = x.view(dim0,-1)
        # x = self.softmax(x)
        l = self.loss(x, labels)
        return MultipleChoiceModelOutput(loss=l, logits=x)
