import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from mymodel.nets.mobilenet import mobilenet_v2

def vgg_model(x):
	vgg = nets.vgg.vgg_19
	my_model = vgg(x)

def inception_model(x):
	inception = nets.inception.inception_v3
	my_model = inception(x)

def resnet_model(x):
	resnet = nets.resnet_v1.resnet_v1_50
	my_model = resnet(x)

def mobile_network(x):
	with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
		logits, endpoints = mobilenet_v2.mobilenet(x)


