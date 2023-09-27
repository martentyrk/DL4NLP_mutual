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
    
    def forward(self, x, y):
        x = self.root_model(x)
        x = self.classifier(x)
        x = x.squeeze(-1)
        x = self.softmax(x)
        l = loss(x, y)
        return {"loss":l, "logits":x}