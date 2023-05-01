import numpy as np
from PIL import Image

def display(frame,shape):
    alter = (frame*255).astype('uint8')
    alter.shape = shape
    img = Image.fromarray(alter).convert('L')
    img.show()

def sigmoid(n:np.ndarray):
    y = n.copy()
    y[n >= 0] = 1.0 / (1 + np.exp(-n[n>=0]))
    y[n < 0 ] = np.exp(n[n<0]) / (1+np.exp(n[n<0]))
    return y

train = np.fromfile('data/train.bin',dtype=np.uint8)/255.0
train_ans = np.fromfile('data/train_ans.bin',dtype=np.uint8)
test  = np.fromfile('data/test.bin',dtype=np.uint8)/255.0
test_ans = np.fromfile('data/test_ans.bin',dtype=np.uint8)

train.shape = (60000,1,784)
test.shape = (10000,1,784)

w1 = np.random.random((784,128))
w1 = (w1*2-1)/784
o1 = np.random.random((1,128))
o1 = (o1*2-1)

w2 = np.random.random((128,10))
w2 = (w2*2-1)/128
o2 = np.random.random((1,10))
o2 = (o2*2-1)

def expect(n):
    return np.array([[0.01]*n+[0.99]+[0.01]*(9-n)])

def loss(result:np.ndarray,pattern:np.ndarray):
    return ((result - pattern)**2)

def score():
    total_loss = 0
    for frame , ans in zip(test,test_ans):
        mid = frame @ w1 + o1
        out = sigmoid( mid @ w2 + o2 )
        pro = loss(out,expect(ans))
        total_loss += pro.sum()
    print('total_loss:',total_loss)

def corate():
    length = 200
    count  = 0 

    for frame,ans in zip(test[:length],test_ans[:length]):
        mid = frame @ w1 + o1
        out = sigmoid( mid @ w2 + o2)
        x = out.argmax()
        if x==ans:count += 1

    # print('correct rate:',count/length)
    return count/length

def trate():
    length = 200
    count  = 0 

    for frame,ans in zip(train[:length],train_ans[:length]):
        mid = frame @ w1 + o1
        out = sigmoid( mid @ w2 + o2)
        x = out.argmax()
        if x==ans:count += 1

    # print('correct rate:',count/length)
    return count/length

step = 0.01

# frame = train[5]
# ans = train_ans[5]
single_train = 1
content_length = 200
wrange = (-1,1)
giveup = 0.5
align  = True

score()

if align:
    max_corate = corate()
else:
    max_corate = trate()
turns = ''
while True:
    turns = input('>')
    if turns=='exit':break
    elif turns=='b':
        content_length+=200
        continue
    elif turns=='a':
        step = float(input('a:'))
        continue
    elif turns=='w':
        wrange = eval(input('w:'))
        continue
    elif turns=='s':
        single_train = int(input('s:'))
        continue
    elif turns=='g':
        giveup = float(input('g:'))
        continue
    elif turns=='t':
        align = not align
        continue
    elif not turns.isdigit():
        continue

    for t in range(int(turns)):

        if align:
            now = corate()
        else:
            now = trate()
        print(t,now)
        if max_corate<now:
            max_corate = now
            w1t = w1.copy()
            w2t = w2.copy()
            o1t = o1.copy()
            o2t = o2.copy()

        for i,(frame,ans) in enumerate(zip(train[:content_length],train_ans[:content_length])):
            for _ in range(single_train):
                mid = frame @ w1 + o1
                out = sigmoid( mid @ w2 + o2 )
                pro = loss(out,expect(ans))

                # print(pro.sum())
                # print(out,out.argmax(),ans)
                if pro.sum() < giveup :break

                offset = expect(ans)-out
                adjust = offset * out * (1-out) * step
                adjust = mid.T @ adjust

                cache = w2+adjust

                if cache.max()<wrange[1] and cache.min()>wrange[0]:
                    w2 = cache
                else:
                    break

                noff = offset @ w2.T
                adjust = noff * mid * (1-mid) * step
                adjust = frame.T @ adjust

                cache = w1+adjust
                if cache.max()<wrange[1] and cache.min()>wrange[0]:
                    w1 = cache
                else:
                    break

    score()
    print('max corate',max_corate)

w1t.tofile('net/w1.bin')
o1t.tofile('net/o1.bin')
w2t.tofile('net/w2.bin')
o2t.tofile('net/o2.bin')