import torch.nn as nn
import torch.nn.functional as F



class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x) # 64 21 4 4
		x = self.leaky_relu(x) # 64
		x = self.conv2(x) # 128 64 4 4
		x = self.leaky_relu(x) # 128
		x = self.conv3(x) # 256 128 4 4
		x = self.leaky_relu(x) # 256
		x = self.conv4(x) # 512 256 4 4
		x = self.leaky_relu(x) # 512
		x = self.classifier(x) # 1 512 4 4
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x # 1
