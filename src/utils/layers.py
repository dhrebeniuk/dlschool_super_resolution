import torch


def crop(image, new_shape):
    width_image = image.shape[2]
    height_image = image.shape[3]
    width_shape = new_shape[2]
    height_shape = new_shape[3]
    start_x = int((width_image - width_shape) / 2)
    start_y = int((height_image - height_shape) / 2)
    cropped_image = image[:, :, start_x:start_x + width_shape, start_y: start_y + height_shape]
    return cropped_image


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
