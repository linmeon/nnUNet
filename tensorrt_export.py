import torch
from monai.networks.nets import UNet




# 初始化模型
model =  UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

# 创建模型的输入数据 元组形式
input = torch.rand(1,1, 64, 64, 64)  # float32
# 导出模型
torch.onnx.export(model,
                  input,
                  r"model.onnx",
                  export_params=True,
                  opset_version=16,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['pred'],
                  )
