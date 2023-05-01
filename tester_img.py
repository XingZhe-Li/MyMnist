import numpy as np
from PIL import Image

net_name = 'net-0.805'

def sigmoid(n:np.ndarray):
    y = n.copy()
    y[n >= 0] = 1.0 / (1 + np.exp(-n[n>=0]))
    y[n < 0 ] = np.exp(n[n<0]) / (1+np.exp(n[n<0]))
    return y

w1 = np.fromfile(f'{net_name}/w1.bin',dtype=np.float64)
o1 = np.fromfile(f'{net_name}/o1.bin',dtype=np.float64)
w2 = np.fromfile(f'{net_name}/w2.bin',dtype=np.float64)
o2 = np.fromfile(f'{net_name}/o2.bin',dtype=np.float64)

w1.shape = (784,128)
o1.shape = (1,128)
w2.shape = (128,10)
o2.shape = (1,10)

print('model loaded')

while True:
    img = Image.open('pic.jpg').convert('L')
    frame = (np.array(img)/255.0).flatten()
    img.close()

    mid = frame @ w1 + o1
    out = sigmoid(mid @ w2 + o2)

    print(out)
    input(out.argmax())