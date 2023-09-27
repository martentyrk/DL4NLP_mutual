class TOD(nn.Module):
    """
    Customized TOD Bert 
    """
    def __init__(self, root_model, num_classes):
        super(CustomModel, self).__init__()
        self.root_model = root_model
        self.classifier = nn.Linear(root_model.config.hidden_size, num_classes, bias=True)
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.root_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.classifier(x)
        x = x.squeeze(-1)
        x = self.softmax(x)
        l = loss(x, labels)
        return {"loss":l, "logits":x}