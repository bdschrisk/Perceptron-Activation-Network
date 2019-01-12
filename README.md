# Perceptron-Activation-Network
A teacher-student activation layer model based on perceptrons, and implemented in PyTorch

### Overview
Typically in deeplearning and neural networks, activation layers are applied to the output of a given layer, whether that layer is a linear, a convolutional layer or otherwise. If we take models from the biological brain we understand that not all parts and pathways of the brain are activated for a given stimulus.  This excitatory process of activation allows only relevant neurons to fire, channeling the knowledge flow and blocks neurons that specialise in other concepts from participating.  This allows sub-networks to develop that are fine-tuned on certain topics or concepts.

### The Perceptron Activation Layer
In this notebook I present a model based on the biological brain, that combines Dropout, ReLU and Perceptron learning into a single learning algorithm - dubbed the Perceptron Activation Layer.  It is used like a typical Linear layer, however internally it implements a teacher-student signal for interrupting the flow of weights that do not contribute to the learning task.  A teacher network is responsible for computing a "teaching signal" that informs the "student network" which weights to activate and which to turn off.

![Perceptron Activation Network](/images/network.png)
_Network (A) standard neural network with activations computed at the nodes.  Each weight forwards the output of the previous node at the given weight value. In contrast with Network (B) which computes an activation of the individual weights, which determines the active weights that are forwarding the input signal to the node.  Additionally, another activation function can be applied in Network (B) at the node for nonlinearity._

NOTES: In the notebook I have used a standard LeNet5 model and replaced the two Linear layers with a 'Perceptron layer' - which achieves ~90% accuracy after the first epoch.  A standard LeNet5 model has also been trained as a comparison, which results in only 20% accuracy after the first epoch.

#### The Algorithm

```

class Perceptron(nn.Module):
    def __init__(self, inputs, outputs, minimum = 0.0):
        super(Perceptron, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        
        # student network
        self.Sb = Parameter(torch.ones(self.outputs, requires_grad=True))
        self.SW = Parameter(torch.randn(self.outputs, self.inputs, requires_grad=True))
        
        # teacher network
        self.S_o, self.S_i = self.SW.size()
        self.Tb = Parameter(torch.ones(self.S_o * self.S_i, requires_grad=True))
        self.TW = Parameter(torch.randn(self.S_o * self.S_i, self.inputs, requires_grad=True))
        
        self.min_val = torch.tensor(minimum).float()
        
        
    def forward(self, x):
        # teacher signal
        self.Tz = F.linear(x, self.TW, self.Tb)
        self.Th = torch.matmul(self.Tz.t(), torch.ones(x.size()[0]))
        
        # perceptron gate
        self.o = torch.gt(self.Th, self.min_val).float()
        
        self.o = self.o.view(self.S_o, self.S_i)
        self.W = torch.mul(self.o, self.SW)
        
        # student signal
        x = F.linear(x, self.W, self.Sb)
        
        return x
```
**
---

@Article{,
  title        = {Perceptron Activation Network},
  author       = {{Chris Kalle}},
  organization = {R4 Robotics Pty Ltd},
  address      = {Queensland, Australia},
  year         = 2018,
  url          = {https://github.com/bdschris/Perceptron-Activation-Network}
}