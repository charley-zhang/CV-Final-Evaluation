import torch, torchvision
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models, transforms


Model1_path = ''
Model2_path = ''
Model3_path = ''
Model4_path = ''
Model5_path = ''
Model6_path = ''
Model7_path = ''
Model8_path = ''




class Model1(nn.Module):

    def __init__(self):
        # ### # Initialize model
        model = models.resnet50(pretrained=False)
        # ### Load state_dict
        state_dict = torch.load(Model1_path)
        model.load_state_dict(state_dict)
        # model.eval()
        #self.model1 = torch.load(Model1_path)['state_dict']
        self.model1 = model

    def __call__(self, X):
        model1 = self.model1
        model1.eval()
        return model1(X)

class Model2(nn.Module):

    def __init__(self):
        # ### # Initialize model
        model = models.resnet50(pretrained=False)
        # ### Load state_dict
        state_dict = torch.load(Model2_path)
        model.load_state_dict(state_dict)
        # model.eval()
        #self.model1 = torch.load(Model1_path)['state_dict']
        self.model2 = model

    def __call__(self, X):
        model2 = self.model2
        model2.eval()
        return model2(X)

class Model3(nn.Module):

    def __init__(self):
        # ### # Initialize model
        model = models.resnet50(pretrained=False)
        # ### Load state_dict
        state_dict = torch.load(Model3_path)
        model.load_state_dict(state_dict)
        # model.eval()
        #self.model1 = torch.load(Model1_path)['state_dict']
        self.model3 = model

    def __call__(self, X):
        model3 = self.model3
        model3.eval()
        return model3(X)


class Model4(nn.Module):

    def __init__(self):
        # ### # Initialize model
        model = models.resnet50(pretrained=False)
        # ### Load state_dict
        state_dict = torch.load(Model4_path)
        model.load_state_dict(state_dict)
        # model.eval()
        #self.model1 = torch.load(Model1_path)['state_dict']
        self.model4 = model

    def __call__(self, X):
        model4 = self.model4
        model4.eval()
        return model4(X)

class Model5(nn.Module):

    def __init__(self):
        # ### # Initialize model
        model = models.resnet50(pretrained=False)
        # ### Load state_dict
        state_dict = torch.load(Model5_path)
        model.load_state_dict(state_dict)
        # model.eval()
        #self.model1 = torch.load(Model1_path)['state_dict']
        self.model5 = model

    def __call__(self, X):
        model5 = self.model5
        model5.eval()
        return model5(X)

class Model6(nn.Module):

    def __init__(self):
        # ### # Initialize model
        model = models.resnet50(pretrained=False)
        # ### Load state_dict
        state_dict = torch.load(Model6_path)
        model.load_state_dict(state_dict)
        # model.eval()
        #self.model1 = torch.load(Model1_path)['state_dict']
        self.model6 = model

    def __call__(self, X):
        model6 = self.model6
        model6.eval()
        return model6(X)

class Model7(nn.Module):

    def __init__(self):
        # ### # Initialize model
        model = models.resnet50(pretrained=False)
        # ### Load state_dict
        state_dict = torch.load(Model7_path)
        model.load_state_dict(state_dict)
        # model.eval()
        #self.model1 = torch.load(Model1_path)['state_dict']
        self.model7 = model

    def __call__(self, X):
        model7 = self.model7
        model7.eval()
        return model7(X)


class Model8(nn.Module):

    def __init__(self):
        # ### # Initialize model
        model = models.resnet50(pretrained=False)
        # ### Load state_dict
        state_dict = torch.load(Model8_path)
        model.load_state_dict(state_dict)
        # model.eval()
        #self.model1 = torch.load(Model1_path)['state_dict']
        self.model8 = model

    def __call__(self, X):
        model8 = self.model8
        model8.eval()
        return model8(X)

class Model9(nn.Module):

    def __init__(self):
        # ### # Initialize model
        model = models.resnet50(pretrained=False)
        # ### Load state_dict
        state_dict = torch.load(Model8_path)
        model.load_state_dict(state_dict)
        # model.eval()
        #self.model1 = torch.load(Model1_path)['state_dict']
        self.model9 = model

    def __call__(self, X):
        model9 = self.model9
        model9.eval()
        return model9(X)








