import numpy as np

net_name = 'net'

def sigmoid(n:np.ndarray):
    y = n.copy()
    y[n >= 0] = 1.0 / (1 + np.exp(-n[n>=0]))
    y[n < 0 ] = np.exp(n[n<0]) / (1+np.exp(n[n<0]))
    return y

w1 = np.fromfile(f'{net_name}/w1.bin',dtype=np.float64)
o1 = np.fromfile(f'{net_name}/o1.bin',dtype=np.float64)
w2 = np.fromfile(f'{net_name}/w2.bin',dtype=np.float64)
o2 = np.fromfile(f'{net_name}/o2.bin',dtype=np.float64)

train = np.fromfile('data/train.bin',dtype=np.uint8)/255.0
train_ans = np.fromfile('data/train_ans.bin',dtype=np.uint8)
test  = np.fromfile('data/test.bin',dtype=np.uint8)/255.0
test_ans = np.fromfile('data/test_ans.bin',dtype=np.uint8)

train.shape = (60000,1,784)
test.shape = (10000,1,784)


w1.shape = (784,128)
o1.shape = (1,128)
w2.shape = (128,10)
o2.shape = (1,10)

print('model loaded')

length = 200
count  = 0 

for frame,ans in zip(test[:length],test_ans[:length]):
    mid = frame @ w1 + o1
    out = sigmoid( mid @ w2 + o2)
    x = out.argmax()
    if x==ans:count += 1

print('correct rate:',count/length)

frame = train[0]
ans   = train_ans[0]

mid = frame @ w1 + o1
out = sigmoid( mid @ w2 + o2)
x = out.argmax()
print(out,x)