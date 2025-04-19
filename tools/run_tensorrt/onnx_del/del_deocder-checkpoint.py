import onnx
from onnx import helper

def onnx_layer_connect(layer_a_name,layer_b_name,graph,i):
    layer_a_output = None
    for node in graph.node:
        if node.name == layer_a_name:
            layer_a_output = node.output[0]  # 获取 Layer A 的输出名称
            break
    if layer_a_output is None:
        print(f"未找到节点 {layer_a_name}")
    else:
        print(f"Layer A 的输出是 {layer_a_output}")
    for node in graph.node:
        if node.name == layer_b_name:
            print(f"修改前 Layer B 的输入是 {node.input}")
            node.input[i] = layer_a_output  # 将 Layer_B 的第一个输入修改为 Layer_A 的输出
            print(f"修改后 Layer B 的输入是 {node.input}")
    # if layer_a_output not in [input.name for input in graph.input]:
    #     new_input = helper.make_tensor_value_info(layer_a_output, onnx.TensorProto.FLOAT, [1, 256, 64, 64])  # 这里的形状需与实际输入匹配
    #     graph.input.append(new_input)

onnx_model = onnx.load('/gemini/code/Sam2-collection/sam2_tensorrt/onnx/conver_tiny_decoder.onnx')
graph = onnx_model.graph

a="/Unsqueeze_11"
b="/Shape_19"
c="/Reshape_5"
onnx_layer_connect(a,b,graph,0)
onnx_layer_connect(a,c,graph,0)

nodes_to_delete = ["/Shape_17", "/Gather_7","/Reshape_3","/Concat_11","/OneHot","/Shape_18","/Slice_2","/Concat_12","/Reshape_4","/Tile","/Gather_29","/Gather_28","/Unsqueeze_31","/Unsqueeze_30","/Concat_23","/Concat_22","/Cast_11","/Shape_36","/Slice_11","/Resize"]  


nodes = graph.node
input = graph.input
output = graph.output
 
print("input\n", input)
print("output\n", output)

for name in nodes_to_delete:
    for node in nodes:
        if name==node.name:
            print(node)
            nodes.remove(node)
######################
for node in graph.node:
    if node.name == '/Clip':
        # 修改节点的输出名称为 'masks'
        node.output[0] = 'masks'
        break
else:
    raise ValueError("未找到名为 '/Clip' 的节点")



onnx.checker.check_model(onnx_model)

onnx.save(onnx_model, '/gemini/code/Sam2-collection/sam2_tensorrt/onnx/conver_tiny_decoder_del.onnx')

print("指定节点已删除")


#./trtexec --onnx=/gemini/code/Sam2-collection/sam2_tensorrt/onnx/conver_tiny_encoder.onnx --saveEngine=/gemini/code/Sam2-collection/onnx/tensorrt/conver_tiny_encoder.engine --int8
#export LD_LIBRARY_PATH=/home/TensorRT-8.6.1.6/lib:LD_LIBRARY_PATH\

# trtexec \
#   --loadEngine=gemini/code/Sam2-collection/sam2_tensorrt/tensorrt/conver_tiny_encoder.engine \
#   --iterations=100 \
#   --warmUp=10 \
#   --verbose \
#   --reportLayerTime