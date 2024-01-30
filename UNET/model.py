import torch 
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        #First convolution layer
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        #Second convolution layer. Number of input channel = out_c coz it would take the output of the RELU() layer which is 64 in this case 64. ([2, 64, 128,128])
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        """Input to the first convolutional layer is the given input""" 
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        """Input to the 2nd Convolutional layer becomes the output of the 1st Conv. layer."""
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

       # print(x.shape) #Prints the torch.Size([2, 64, 128,128]) where 2 = batch size, 64 = input channel, 128 *128 is size
        return x

"""Encoder block is basically a convolutional block followed by the pooling layer i.e maxpooling
    Basically, the spatial dimention i.e height and width are reduced as we go from encoder1, encoder2 ...
    And Number of filter i.e 64 increases to 128, 256 as we go 
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        #First the convolutional block followed by a 2*2 pooling layer 
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        """First input goes through the convolutional layer and then to the pooling layer 
        Returns the x- output of convolutional block and y - output of pooling block """
        x = self.conv(inputs) # output of this convolution block act as skip conncetion for the decorder
        p = self.pool(x)

        return x, p

#Define the decoder block : Transpose convolutional -> skip connection -> followed by convolution block
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        """It starts with 2*2 transpose convolutional. Takes input channel, output channel
and since we need to upsample (increase) features height and width by stride of 2  """
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    """ For the first decorder block, it takes first input as the output of the 
    bottleneck and skip conncetion """
    def forward(self, inputs, skip):
        x = self.up(inputs)
        #print(x.shape) #It will have the shape of the upsampled feature map i.e : torch.Size([2,512,64,64])
        """For this we need the skip connection with the same shape as x. Turns out the 4th skip conncection has the same shape
        Then concationation with the skip connection helps to capture information from encorder to decorder. Helps in better generation
        of the feature map."""
        x = torch.cat([x, skip], axis = 1)
        x = self.conv(x)
        return x
    
    """Later the feature map formed are used to generation of segmentation map or binary mask"""

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64) # Input channel for e1 is 3 (RGB image) and output channel is 64 
        self.e2 = encoder_block(64, 128) #Input chaneel is 64 and output channel is 128
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck or the Bridge layer b/w encoder and decorder which is 
         basically the convolution block """
        self.b = conv_block(512, 1024)

        """Decorder: Here the height and width increases by 2 and output channels-> feature map decreases by 2"""
        self.d1 = decoder_block(1024, 512) #Input channel, output channel
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier: To generate the seg map. 1*1 conv. map and we need 1 channel as its a binary seg. problem."""
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)


    def forward(self, inputs):
        """ Encoder: Four skip conncetions and 4 pooling layer """
        s1, p1 = self.e1(inputs) # encoder block returns skip connection and pooling output
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)
        
        #Confirming the shapes
        # print(s1.shape, s2.shape, s3.shape, s4.shape)
        # print(b.shape)
        """Outputs :
        torch.Size([2, 64, 512, 512]) torch.Size([2, 128, 256, 256]) torch.Size([2, 256, 128, 128]) torch.Size([2, 512, 64, 64])
        torch.Size([2, 1024, 32, 32]) 

        Note how the dimentions are decreasing while the feature map is increasing in the output."""
        
        """Decorder"""
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # print(d4.shape) # Prints the same shape as the input i.e feature map = 64 with 512*512 as height and width
        # # Output : torch.Size([2, 64, 512, 512])

        outputs = self.outputs(d4)

        return outputs


if __name__ == "__main__":
    
    """Testing the convolutional block class"""
    # x = torch.randn((2, 32, 128, 128))
    #f = conv_block(32, 64) # 32 = input size, 64 = output size
    #y = f(x) #Input x with output y 
    #print(y.shape) # Output = torch.Size([2, 64, 128, 128])

    """Now with the encoder_block, when we send x as input, we get y and p 
    as the output. 
    y is the output of convolutional block whose size(height*width) is 128*128 while
    p is feature map where the size reduces to 64*64 as we have used the maxpooling of 2*2 matrix.
    """
    """Testing encoder block class"""
    # x = torch.randn((2, 32, 128, 128))
    # f = encoder_block(32, 64) # 32 = input size, 64 = output size
    # y, p = f(x) 
    # print(y.shape) # Outputs : torch.Size([2, 64, 128, 128])
    # print(p.shape) # Outputs : torch.Size([2, 64, 64, 64])

    """Testing the build_unet class to see the output from the classifier : binary segmentation"""
    x = torch.randn((2, 3, 512, 512))
    f = build_unet()
    y = f(x)
    print(y.shape) #torch.Size([2, 1, 512, 512]) which means batch size of 2 and no. of channels as 1 (binary mask)




