import numpy as np
filename = 'D:/1TJU/AAcode/practice2/Luna/lunaset/preprocess/035_clean.npy'
imgs= np.load(filename)
# target = [90.6355361938476,83.7471542358398,41.6657371520996,10.465853691101]
# target = [97.21,133.55,289.41,10.0695]
target = [175.44,149.73,103.63,5.916]
bound_size = 3
crop_size = [36,36,36]

# start = []
# for i in range(3):
#     r = target[3] / 2
#     s = np.floor(target[i] - r) + 1 - bound_size  #np.floor 返回不大于输入参数的最大整数
#     e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i] #np.ceil 函数返回输入值的上限
#     if s > e:
#         start.append(np.random.randint(e, s))  # !
#     else:
#         start.append(int(target[i]) - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2))
# pad = []
# pad.append([0, 0])
# for i in range(3):
#     leftpad = max(0, -start[i])
#     rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
#     pad.append([leftpad, rightpad])
# crop = imgs[:,
#        max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
#        max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
#        max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]
# crop = np.pad(crop, pad, 'constant', constant_values=0)
#
# #############################
def crop( imgs, target, train=True):
    crop_size = [36,36,36]
    bound_size = 3
    target = np.copy(target)

    start = []
    for i in range(3):
        start.append(int(target[i]) - int(crop_size[i] / 2))

    pad = []
    pad.append([0, 0])
    for i in range(3):
        leftpad = max(0, -start[i])
        rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
        pad.append([leftpad, rightpad])
    crop = imgs[:,
           max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
           max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
           max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]

    crop = np.pad(crop, pad, 'constant', constant_values=0)

    for i in range(3):
        target[i] = target[i] - start[i]

    return crop, target


# def __call__(self, imgs, target, train=True):
crop_img, target = crop(imgs, target,True)
imgs = np.squeeze(crop_img, axis=0)

z = int(target[0])
y = int(target[1])
x = int(target[2])
# z = 24
# y = 24
# x = 24

nodule_size = int(target[3])
margin = max(7, nodule_size * 0.4)
radius = int((nodule_size + margin) / 2)

s_z_pad = 0
e_z_pad = 0
s_y_pad = 0
e_y_pad = 0
s_x_pad = 0
e_x_pad = 0

s_z = max(0, z - radius)
if (s_z == 0):
    s_z_pad = -(z - radius)

e_z = min(np.shape(imgs)[0], z + radius)
if (e_z == np.shape(imgs)[0]):
    e_z_pad = (z + radius) - np.shape(imgs)[0]

s_y = max(0, y - radius)
if (s_y == 0):
    s_y_pad = -(y - radius)

e_y = min(np.shape(imgs)[1], y + radius)
if (e_y == np.shape(imgs)[1]):
    e_y_pad = (y + radius) - np.shape(imgs)[1]

s_x = max(0, x - radius)
if (s_x == 0):
    s_x_pad = -(x - radius)

e_x = min(np.shape(imgs)[2], x + radius)
if (e_x == np.shape(imgs)[2]):
    e_x_pad = (x + radius) - np.shape(imgs)[2]

# print (s_x, e_x, s_y, e_y, s_z, e_z)
# print (np.shape(img_arr[s_z:e_z, s_y:e_y, s_x:e_x]))
nodule_img = imgs[s_z:e_z, s_y:e_y, s_x:e_x]
nodule_img = np.pad(nodule_img, [[s_z_pad, e_z_pad], [s_y_pad, e_y_pad], [s_x_pad, e_x_pad]], 'constant',
                    constant_values=0)

imgpad_size = [36 - np.shape(nodule_img)[0],
               36 - np.shape(nodule_img)[1],
               36 - np.shape(nodule_img)[2]]
imgpad = []
imgpad_left = [int(imgpad_size[0] / 2),
               int(imgpad_size[1] / 2),
               int(imgpad_size[2] / 2)]
imgpad_right = [int(imgpad_size[0] / 2),
                int(imgpad_size[1] / 2),
                int(imgpad_size[2] / 2)]

for i in range(3):
    if (imgpad_size[i] % 2 != 0):

        rand = np.random.randint(2)
        if rand == 0:
            imgpad.append([imgpad_left[i], imgpad_right[i] + 1])
        else:
            imgpad.append([imgpad_left[i] + 1, imgpad_right[i]])
    else:
        imgpad.append([imgpad_left[i], imgpad_right[i]])

padding_crop = np.pad(nodule_img, imgpad, 'constant', constant_values=0)

padding_crop = np.expand_dims(padding_crop, axis=0)

crop = np.concatenate((padding_crop, crop_img))


from matplotlib import pyplot as plt

def plot_nodule(nodule_crop):
    # Learned from ArnavJain
    # https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    f, plots = plt.subplots(int(nodule_crop.shape[0] / 4) + 1, 4, figsize=(10, 10))

    for z_ in range(nodule_crop.shape[0]):
        plots[int(z_ / 4), z_ % 4].imshow(nodule_crop[z_, :, :])
        # plots[int(z_ / 4), z_ % 4].imshow(nodule_crop[z_, :, :], cmap='gray')

    # The last subplot has no image because there are only 19 images.
    plt.show()
plot_nodule(crop[0])