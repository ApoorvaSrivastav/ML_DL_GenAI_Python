import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = torch.tensor(1.43)
        self.b1 = torch.tensor(-0.61)
        self.W2 = torch.tensor(2.63)
        self.b2 = torch.tensor(1.35)
        self.W3 = torch.tensor(-3.89)
        self.W4 = torch.tensor(1.35)

    def forward(self,x):
        y1 =  (x*self.W1) + self.b1    
        y2 = x* self.W2 + self.b2
        y3  = F.relu(y1)
        y4 = F.relu(y2)
        y5 = y3*self.W3 + y4*self.W4  
        return y5 
    
if __name__ == "__main__":
    model = NeuralNet()
    input_val = torch.tensor([0.0,0.3,0.6,1.0])
    output = model(input_val)
    print(output)

    #Creating graph for each value calculated within the model
    doses = torch.linspace(0,1,20)
    hiddenlayer_node1_input = doses * model.W1 +model.b1
    hiddenlayer_node2_input = doses * model.W2 + model.b2
    hiddenlayer_node1_output = F.relu(hiddenlayer_node1_input)
    hiddenlayer_node2_output = F.relu(hiddenlayer_node2_input)
    Final_node_input1 = hiddenlayer_node1_output * model.W3
    Final_node_input2 = hiddenlayer_node2_output * model.W4
    output = hiddenlayer_node1_output * model.W3 + hiddenlayer_node2_output * model.W4

    sns.set(style = "whitegrid")

    #graph for each intermediate output

    sns.scatterplot(x = doses, y = hiddenlayer_node1_input, color = "red")
    sns.lineplot(x = doses, y= hiddenlayer_node1_input, label = "Hidden Layer Node 1 Input Line", color = "blue")

    sns.scatterplot(x = doses, y = hiddenlayer_node2_input,  color = "green")
    sns.lineplot(x = doses, y= hiddenlayer_node2_input, label = "Hidden Layer Node 2 Input Line", color = "orange")

    sns.scatterplot(x = doses, y = hiddenlayer_node1_output,color = "purple")
    sns.lineplot(x = doses, y= hiddenlayer_node1_output, label = "Hidden Layer Node 1 Output Line", color = "brown")

    sns.scatterplot(x = doses, y = hiddenlayer_node2_output, color = "cyan")
    sns.lineplot(x = doses, y= hiddenlayer_node2_output, label = "Hidden Layer Node 2 Output Line", color = "magenta")

    sns.scatterplot(x = doses, y = Final_node_input1, color = "yellow")
    sns.lineplot(x = doses, y= Final_node_input1, label = "Final Node 1 Input Line", color = "gold",linestyle = "--")
    sns.scatterplot(x = doses, y = Final_node_input2, color = "darkgreen")
    sns.lineplot(x = doses, y= Final_node_input2, label = "Final Node 2 Input Line", color = "darkblue",linestyle = "--")

    sns.scatterplot(x = doses, y = output, color = "black")
    sns.lineplot(x = doses, y= output, label = "Output Line", color = "gray",linestyle = ":")

    plt.xlabel("Dose")
    plt.ylabel("Value")
    plt.title("Neural Network Intermediate Values")
    plt.show()