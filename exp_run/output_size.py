import numpy as np


def calculation(input, kernel, pad, stride):
    """
    Calculate the output dimensions of an input passing through a
    convolutional layer

    Inputs
        input: numpy.array - The input array
        kernel: numpy.array - The kernel size for the filter
        pad: numpy.array - The amount of zero padding on the input
        stride: numpy.array - The gap between filters passing over the input

    Output
        dim: numpy.array - The output dimensions
    """

    dim = ((input - (kernel - 1) + (2 * pad)) - 1) / stride
    dim += 1
    return dim


def dim_max_pool(dim):
    """
    Determine the potential values to be used for a max pooling layer
    """
    ref0 = int(dim[0])
    ref1 = int(dim[1])
    nums0 = []
    nums1 = []
    for i in list(range(1, ref0)):
        if ref0 % i == 0:
            nums0.append(i)
    print('Acceptable values for max pool: ', nums0)
    for i in list(range(1, ref1)):
        if ref1 % i == 0:
            nums1.append(i)
    print('Acceptable values for max pool: ', nums1)


def calculation_gru(dim, hidden_size, bidi):
    """
    input: frequency, time
    output: batch, time, featuremap
    however here we are not concerned with batch
    """
    dim = dim[1]
    if bidi:
        value = 2
    else:
        value = 1
    fm = hidden_size * value
    dim = np.array([dim, fm])

    return dim


def determine_stride_pad(input_dim, kernel, pad, stride, rng):
    """
    Calculate possible values for the stride and padding given some input and
    a kernel filter

    Inputs
        input_dim: numpy.array - The dimensions of the input array
        kernel: numpy.array - The kernel size for the filter
        pad: numpy.array - The amount of zero padding on the input
        stride: numpy.array - The gap between filters passing over the input
        rng: int - The range to test values for stride and padding

    Output
        values: list - Appropriate values for stride and padding, given the
                input dimensions and the kernel size
    """
    values = []
    for i in range(rng):
        for j in range(rng):
            dim = calculation(input_dim, kernel, pad, stride)
            temp = np.round(dim[-1])
            if temp == dim[-1]:
                values.append([stride[-1], pad[-1]])
            stride += 1
        stride = np.array([1, 1])
        pad += 1

    return values


input_dim = np.array([1, 61440])
weights = 0
# Conv 1a
kernel = np.array([1, 1024])
stride = np.array([1, 512])
pad = np.array([0, 0])
dim = calculation(input_dim, kernel, pad, stride)
list_of_values = determine_stride_pad(input_dim, kernel, pad, stride, 512)
for i, d in enumerate(list_of_values):
    if d[0] == 512:
        print(i)

weights = 0
batch = 18
print(dim)

# Conv 1b
kernel = np.array([1, 3])
stride = np.array([1, 1])
pad = np.array([1, 1])
# dim = calculation(dim, kernel, pad, stride)
print(dim)

dim_max_pool(dim)

# Max Pool 1
kernel = np.array([1, 3])
pad = np.array([0, 0])
stride = np.array([1, 3])
dim = calculation(dim, kernel, pad, stride)
print(dim)

# Conv 2a
kernel = np.array([3, 3])
pad = np.array([1, 1])
stride = np.array([1, 1])
# dim = calculation(dim, kernel, pad, stride)
print(dim)

# Conv 2b
kernel = np.array([3, 3])
pad = np.array([1, 1])
stride = np.array([1, 1])
# dim = calculation(dim, kernel, pad, stride)
print(dim)

# Max Pool 2
kernel = np.array([1, 2])
pad = np.array([0, 0])
stride = np.array([1, 2])
# dim = calculation(dim, kernel, pad, stride)
print(dim)

weights = 0
# Conv 3a
kernel = np.array([3, 3])
stride = np.array([1, 1])
pad = np.array([1, 1])
#dim = calculation(dim, kernel, pad, stride)
weights = 0
print(dim)

# Conv 3b
kernel = np.array([3, 3])
stride = np.array([1, 1])
pad = np.array([1, 1])
#dim = calculation(dim, kernel, pad, stride)
print(dim)

dim_max_pool(dim)

# Max Pool 3
kernel = np.array([2, 2])
pad = np.array([0, 0])
stride = np.array([2, 2])
#dim = calculation(dim, kernel, pad, stride)
print(dim)

filter_size = 128

# GRU + FC
bidi = False
hidden_size = 128
dim = calculation_gru(dim, hidden_size, bidi)
print(dim)
print(int(dim[0] * dim[1]))

# print('Global Pooling dimensions: ', dim[0], dim[1])
# For Fully Connected
print(int(dim[0] * dim[1] * filter_size))
# print('If Global_Pool: ', filter_size*batch)
