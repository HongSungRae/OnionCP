import torch.nn as nn
import torch





class ResidualBlock(nn.Module):
  def __init__(self, in_feature):
    super().__init__()

    self.block = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(in_feature, in_feature, 3),
      nn.InstanceNorm2d(in_feature),
      nn.ReLU(inplace=True),
      nn.ReflectionPad2d(1),
      nn.Conv2d(in_feature, in_feature, 3),
      nn.InstanceNorm2d(in_feature),
    )
  
  def forward(self, x):
    return x + self.block(x)




class GenertorResNet(nn.Module):
  def __init__(self, input_shape, num_residual_block):
    super().__init__()

    channels = input_shape[0]

    out_features = 64

    model = [
      nn.ReflectionPad2d(channels),
      nn.Conv2d(channels, out_features, 7),
      nn.InstanceNorm2d(out_features),
      nn.ReLU(inplace=True)
    ]

    in_features = out_features

    for _ in range(2):
      out_features *= 2
      model += [
        nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(out_features),
        nn.ReLU(inplace=True)
      ]
      in_features = out_features
    
    for _ in range(num_residual_block):
      model += [ResidualBlock(out_features)]
    
    for _ in range(2):
      out_features //= 2

      model += [
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_features),
        nn.ReLU(inplace=True)
      ]
      in_features = out_features

    model += [
      nn.ReflectionPad2d(channels),
      nn.Conv2d(out_features, channels, 7),
      nn.Tanh()
    ]

    self.model = nn.Sequential(*model)

  def forward(self, noise, x):
    # x = torch.cat([x,x,x], dim=1)
    return self.model(x)