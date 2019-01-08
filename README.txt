To run:

python prediction_tool.py <network name>

network name:
vgg_19 -> vgg
resnet_v1_50 -> resnet
inception_v3 -> inception
mobile_net_v1 -> mobilenet

1. I've included the spreadsheets and a screenshot with total network values in this folder. 
   The pdf file could not fit columns properly. The program will generate a spreadsheet on its own too.

2. All models have been imported from slim.nets, the code should work for most of other models too. Mobile net has been included
   explicitly in mymodel folder.

3. macs per cycle will be modified in the program based on matrix and vector operations. So, hardware.txt will only require 3 parameters.

4. Please note I have not verified the network values. However, mobile net has least compute and vgg-19 has the most, which is expected.

5. I was able to extract all the parameters for each operation from the graph except max_pooling filter size. This is accounted by using a kernel variable for max_pooling.

