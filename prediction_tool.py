import tensorflow as tf
import numpy as np
from collections import OrderedDict
import re
import pandas as pd
import sys
from utilities import *
from mynetworks import *


x = np.zeros((10,224,224,3)) # Padded input
x_tf = tf.convert_to_tensor(x, np.float32)
display = 1e+06

hardware = {}
with open("hardware.txt") as f:
	for line in f:
		(key, val) = line.split()
		hardware[key] = float(val)


internal_memory = hardware['memory']
bandwidth = hardware['bandwidth']
macs_per_cycle = hardware['macs']


required_operations = ['Conv2D','Relu6','MaxPool','Floor','BiasAdd','FusedBatchNorm','Relu','Mean']

network = sys.argv[1]

if network == "vgg":
	vgg_model(x_tf)
	max_pool_kernel = 2
	stride = 2

if network == "resnet":
	resnet_model(x_tf)
	ave_pool_kernel = 7
	max_pool_kernel = 2
	stride = 2

if network == "inception":
	inception_model(x_tf)
	ave_pool_kernel = 3
	max_pool_kernel = 3
	stride = 2

if network == "mobilenet":
	mobile_network(x_tf)

graph = tf.get_default_graph()
layers = [op.name for op in graph.get_operations() if op.type in required_operations]
layers.insert(0,'Const')

layers_dict = OrderedDict()
filter_dict = OrderedDict()
final_dictionary = OrderedDict()

# List order for each layers is [Number of images, output rows, output columns, output channels]
for name in layers:
    layers_dict[name] = graph.get_tensor_by_name(name+':0').get_shape().as_list()

for op in graph.get_operations():
    if op.type == "VariableV2" and "weights" in op.name:
        filter_dict[op.name] = graph.get_tensor_by_name(op.name+':0').get_shape().as_list()

layers_list_values = list(layers_dict.values())
filter_list_values = list(filter_dict.values())

# Format: [no_of_images, ch_in, dim_in_row, dim_in_column, ch_out, dim_out_row, dim_out_column, filter_row, filter_column]
for key,my_name in enumerate(layers_dict,0):
    if my_name == "Const":
        final_dictionary[my_name] = [layers_list_values[key][0],layers_list_values[key][3],layers_list_values[key][1],layers_list_values[key][2],
                                         layers_list_values[key][3],layers_list_values[key][1],layers_list_values[key][2], 0, 0]    
    if '/' in my_name:
        name_stripped = re.search('.*\\/',my_name).group(0)
        if 'InceptionV3/InceptionV3' in name_stripped:
            name_stripped = name_stripped[12:]
        
    if "Conv2D" in my_name:
        final_dictionary[name_stripped] = [layers_list_values[key-1][0],layers_list_values[key-1][3],layers_list_values[key-1][1],layers_list_values[key-1][2],
                                         layers_list_values[key][3],layers_list_values[key][1],layers_list_values[key][2]] + filter_dict[name_stripped+'weights'][0:2]
    if "Relu" in my_name:
        final_dictionary[my_name] = [layers_list_values[key-1][0],layers_list_values[key-1][3],layers_list_values[key-1][1],layers_list_values[key-1][2],
                                         layers_list_values[key][3],layers_list_values[key][1],layers_list_values[key][2], 0, 0]
    if "dropout" in my_name:
        final_dictionary[name_stripped] = [layers_list_values[key-1][0],layers_list_values[key-1][3],layers_list_values[key-1][1],layers_list_values[key-1][2],
                                         layers_list_values[key][3],layers_list_values[key][1],layers_list_values[key][2], 0, 0]
    if "MaxPool" in my_name:
        final_dictionary[name_stripped] = [layers_list_values[key-1][0],layers_list_values[key-1][3],layers_list_values[key-1][1],layers_list_values[key-1][2],
                                           layers_list_values[key][3],layers_list_values[key][1],layers_list_values[key][2], max_pool_kernel, max_pool_kernel]
    if "BiasAdd" in my_name:
        final_dictionary[my_name] = [layers_list_values[key-1][0],layers_list_values[key-1][3],layers_list_values[key-1][1],layers_list_values[key-1][2],
                                         layers_list_values[key][3],layers_list_values[key][1],layers_list_values[key][2], 0, 0]
    if "Relu6" in my_name:
        final_dictionary[my_name] = [layers_list_values[key-1][0],layers_list_values[key-1][3],layers_list_values[key-1][1],layers_list_values[key-1][2],
                                         layers_list_values[key][3],layers_list_values[key][1],layers_list_values[key][2], 0, 0]
    if "FusedBatchNorm" in my_name:
        final_dictionary[my_name] = [layers_list_values[key-1][0],layers_list_values[key-1][3],layers_list_values[key-1][1],layers_list_values[key-1][2],
                                         layers_list_values[key][3],layers_list_values[key][1],layers_list_values[key][2], 0, 0]
    if "pool" in my_name and "Max" not in my_name:
        final_dictionary[name_stripped] = [layers_list_values[key-1][0],layers_list_values[key-1][3],layers_list_values[key-1][1],layers_list_values[key-1][2],
                                         layers_list_values[key][3],layers_list_values[key][1],layers_list_values[key][2], ave_pool_kernel, ave_pool_kernel] 


for key, value in final_dictionary.items():
        final_dictionary[key]+=[feature_map_memory(value[1],value[4],value[2],value[3]),filter_coefficient_memory(value[1],value[4],value[7],value[8])
                               ,macs(value[2],value[3],value[1],value[4],value[7],value[8]),comp_relu_pool(value[1],value[2],value[3]),activation(value[4],value[5],value[6])]
        

# Calculating bytes and memory location
# memory location 0 -> internal
# 1 -> external (data movement required)
for key, value in final_dictionary.items():
    temp_input_bytes = get_bytes(value[1],value[2],value[3])
    temp_output_bytes = get_bytes(value[4],value[5],value[6])
    temp_filter_coeff_bytes = (value[10]*bits_per)/8
    temp_memory = internal_memory
    assign_input = 0
    assign_output = 0
    
    if temp_input_bytes < temp_memory:
        assign_input = 0
        temp_memory -= temp_input_bytes
    else:
        assign_input = 1
    
    if temp_output_bytes < temp_memory:
        assign_output = 0
        temp_memory -= temp_output_bytes
    else:
        assign_output = 1
    
    final_dictionary[key]+=[temp_input_bytes,temp_output_bytes,temp_filter_coeff_bytes,assign_input, assign_output,1]

# CNN 2D conv will be matrix-matrix
# Bias and Relu will be free (macs will be 0)
# others will be vector

for key, value in final_dictionary.items():
    temp_input_data_movement = (value[17]*value[14]/bandwidth)*display
    temp_output_data_movement = (value[18]*value[15]/bandwidth)*display
    temp_filter_coeff_data_movement = (value[19]*value[16]/bandwidth)*display
    total_data_movement = temp_input_data_movement + temp_output_data_movement + temp_filter_coeff_data_movement
    if "Conv2d" in key or "conv" in key and "Relu" not in key and "BiasAdd" not in key:
        temp_matrix_compute = (value[11]/macs_per_cycle/value[2])*display
        temp_vector_compute = 0
    else:
        temp_matrix_compute = 0
        temp_vector_compute = (value[11]/macs_per_cycle)*display
    total_compute_time = temp_matrix_compute +temp_vector_compute
    serial_total = total_data_movement + total_compute_time
    parallel_total = max(total_data_movement,total_compute_time)
    final_dictionary[key]+=[temp_input_data_movement,temp_output_data_movement,temp_filter_coeff_data_movement,total_data_movement,
                           temp_matrix_compute,temp_vector_compute,total_compute_time,serial_total,parallel_total]

my_df = pd.DataFrame(data = final_dictionary, dtype = 'float32')
my_df_T = my_df.T
my_df_T.columns = ['no_of_images', 'ch_in', 'dim_in_row', 'dim_in_column', 'ch_out', 'dim_out_row', 'dim_out_column', 'filter_row', 'filter_column',
                   'feature_map_memory','filter_coefficient_memory','macs','comp','activation','input_map_bytes','output_map_bytes','fitler_coeff_bytes'
                  ,'input_map_location','output_map_location','fitler_location','input_data_movement*'+str(display),'output_data_movement*'+str(display),'filter_data_movement*'+str(display),
                  'total_data_movement*'+str(display),'matrix_compute_time*'+str(display),'vector_compute_time*'+str(display),'total_compute_time*'+str(display),'serial*'+str(display),'parallel*'+str(display)]
my_df_T.to_csv(network + '.csv', encoding = 'utf-8')

total_network_values = sum(np.asarray(list(final_dictionary.values()))[:, 20:])

print("Total Values: \n Input feature map data movement time: {} \n Output feature map data movement time: {} \n Filter coefficient data movement time: {}".format(total_network_values[0],total_network_values[1],total_network_values[2]))
print("\n Total data movement time for the network : {} \n Matrix compute time : {} \n Vector compute time: {}".format(total_network_values[3],total_network_values[4],total_network_values[5]))
print("\n Total compute time for the network  : {} \n Sum of serial total data movement: {} \n Sum of parallel total data movement: {}".format(total_network_values[6],total_network_values[7],total_network_values[8]))
