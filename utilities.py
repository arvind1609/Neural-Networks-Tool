images = 1
bits_per = 8
def feature_map_memory (input_channels, output_channels ,image_rows, image_columns):
    return (input_channels+output_channels)*image_rows*image_columns

def filter_coefficient_memory(input_channels, output_channels, filter_rows, filter_columns):
    return (input_channels*output_channels*filter_rows*filter_columns) + output_channels

def macs(image_rows, image_columns, input_channels, output_channels, filter_rows, filter_columns):
	global images
	return ((image_rows*image_columns*input_channels*output_channels*filter_rows*filter_columns))*images

def comp_relu_pool(input_channels, input_rows, input_columns):
	global images
	return (input_channels*input_rows*input_columns)*images   

def activation(output_channels, output_rows, output_columns):
	global images
	return (output_channels*output_rows*output_columns)*images

def get_bytes(channels, rows, columns):
	global bits_per
	return ((channels*rows*columns*bits_per)/8)
