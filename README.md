# TensorFlow1
TensorFlow
记的在学生时代，英语考试有这么一种类型的题，叫：阅读理解。首先让你读一段洋文材料，然后回答一些基于这个洋文材料提的问题。

我先给你出一道阅读理解
Big Panda learned to code when he was 21. He live in China and have no life, feel like a big loser. But here is one thing Panda want you to remember…it´s never too late! You can do anything if you put your heart on it!

____ is the loser.（下划线处该填什么呢？）

我出的这道填空题，对人来说轻而易举，但是要让机器回答就很难了。机器阅读和理解人类语言是非常有挑战性的。

本帖就使用TensorFlow练习一个阅读理解，看看准确率能到什么程度。

使用的数据集
https://research.fb.com/projects/babi/
http://cs.nyu.edu/~kcho/DMQA/
本帖只使用”非死不可”提供的《Children’s Book Test》数据集。

TensorFlow练习25: 使用深度学习做阅读理解+完形填空

数据预处理

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
import re
import random
import ast
import itertools
import pickle
import numpy as np
 
train_data_file = './CBTest/data/cbtest_NE_train.txt'
valid_data_file = './CBTest/data/cbtest_NE_valid_2000ex.txt'
 
def preprocess_data(data_file, out_file):
	# stories[x][0]  tories[x][1]  tories[x][2]
	stories = []
	with open(data_file) as f:
		story = []
		for line in f:
			line = line.strip()
			if not line:
				story = []
			else:
				_, line = line.split(' ', 1)
				if line:
					if '\t' in line:
						q, a, _, answers = line.split('\t')
						# tokenize
						q = [s.strip() for s in re.split('(\W+)+', q) if s.strip()]
						stories.append((story, q, a))
					else:
						line = [s.strip() for s in re.split('(\W+)+', line) if s.strip()]
						story.append(line)
 
	samples = []
	for story in stories:
		story_tmp = []
		content = []
		for c in story[0]:
			content += c
		story_tmp.append(content)
		story_tmp.append(story[1])
		story_tmp.append(story[2])
 
		samples.append(story_tmp)
 
	random.shuffle(samples)
	print(len(samples))
 
	with open(out_file, "w") as f:
		for sample in samples:
			f.write(str(sample))
			f.write('\n')
 
preprocess_data(train_data_file, 'train.data')
preprocess_data(valid_data_file, 'valid.data')
 
# 创建词汇表
def read_data(data_file):
	stories = []
	with open(data_file) as f:
		for line in f:
			line = ast.literal_eval(line.strip())
			stories.append(line)
	return stories
 
 
stories = read_data('train.data') + read_data('valid.data')
 
content_length = max([len(s) for s, _, _ in stories])
question_length = max([len(q) for _, q, _ in stories])
print(content_length, question_length)
 
vocab = sorted(set(itertools.chain(*(story + q + [answer] for story, q, answer in stories))))
vocab_size = len(vocab) + 1
print(vocab_size)
word2idx = dict((w, i + 1) for i,w in enumerate(vocab))
pickle.dump((word2idx, content_length, question_length, vocab_size), open('vocab.data', "wb"))
 
# From keras 补齐
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.):
    lengths = [len(s) for s in sequences]
 
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
 
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
 
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)
 
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
 
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
 
# 转为向量
def to_vector(data_file, output_file):
	word2idx, content_length, question_length, _ = pickle.load(open('vocab.data', "rb"))
	
	X = []
	Q = []
	A = []
	with open(data_file) as f_i:
		for line in f_i:
			line = ast.literal_eval(line.strip())
			x = [word2idx[w] for w in line[0]]
			q = [word2idx[w] for w in line[1]]
			a = [word2idx[line[2]]]
 
			X.append(x)
			Q.append(q)
			A.append(a)
 
	X = pad_sequences(X, content_length)
	Q = pad_sequences(Q, question_length)
 
	with open(output_file, "w") as f_o:
		for i in range(len(X)):
			f_o.write(str([X[i].tolist(), Q[i].tolist(), A[i]]))
			f_o.write('\n')
 
to_vector('train.data', 'train.vec')
to_vector('valid.data', 'valid.vec')
 
 
"""
# to_word
word2idx, content_length, question_length, _ = pickle.load(open('vocab.data', "rb"))
 
def get_value(dic,value):
    for name in dic:
        if dic[name] == value:
            return name
 
with open('train.vec') as f:
	for line in f:
		line = ast.literal_eval(line.strip())
		for word in line[0]:
			print(get_value(word2idx, word))
"""
生成的文件：vocab.data词汇表、train.vec、valid.vec数据的向量表示。

训练
TensorFlow练习25: 使用深度学习做阅读理解+完形填空
https://arxiv.org/pdf/1606.02245v4.pdf

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156


import
tensorflow
as
tf


import
pickle


import
numpy
as
np


import
ast


from
collections
import
defaultdict


 


train_data
=
'train.vec'


valid_data
=
'valid.vec'


 


word2idx,
content_length,
question_length,
vocab_size
=
pickle.load(open('vocab.data',
"rb"))


print(content_length,
question_length,
vocab_size)


 


batch_size
=
64


 


train_file
=
open(train_data)


def
get_next_batch():


X
=
[]


Q
=
[]


A
=
[]


for
i
in
range(batch_size):


for
line
in
train_file:


line
=
ast.literal_eval(line.strip())


X.append(line[0])


Q.append(line[1])


A.append(line[2][0])


break


 


if
len(X)
==
batch_size:


return
X,
Q,
A


else:


train_file.seek(0)


return
get_next_batch()


 


def
get_test_batch():


with
open(valid_data)
as
f:


X
=
[]


Q
=
[]


A
=
[]


for
line
in
f:


line
=
ast.literal_eval(line.strip())


X.append(line[0])


Q.append(line[1])


A.append(line[2][0])


return
X,
Q,
A


 


 


X
=
tf.placeholder(tf.int32,
[batch_size,
content_length])  
#
 洋文材料


Q
=
tf.placeholder(tf.int32,
[batch_size,
question_length])  #
 问题


A
=
tf.placeholder(tf.int32,
[batch_size])                  
#
 答案


 


#
 drop out


keep_prob
=
tf.placeholder(tf.float32)


 


def
glimpse(weights,
bias,
encodings,
inputs):


weights
=
tf.nn.dropout(weights,
keep_prob)


inputs
=
tf.nn.dropout(inputs,
keep_prob)


attention
=
tf.transpose(tf.matmul(weights,
tf.transpose(inputs))
+
bias)


attention
=
tf.batch_matmul(encodings,
tf.expand_dims(attention,
-1))


attention
=
tf.nn.softmax(tf.squeeze(attention,
-1))


return
attention,
tf.reduce_sum(tf.expand_dims(attention,
-1)
*
encodings,
1)


 


def
neural_attention(embedding_dim=384,
encoding_dim=128):


embeddings
=
tf.Variable(tf.random_normal([vocab_size,
embedding_dim],
stddev=0.22),
dtype=tf.float32)


tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4),
[embeddings])


 


with
tf.variable_scope('encode'):


with
tf.variable_scope('X'):


X_lens
=
tf.reduce_sum(tf.sign(tf.abs(X)),
1)


embedded_X
=
tf.nn.embedding_lookup(embeddings,
X)


encoded_X
=
tf.nn.dropout(embedded_X,
keep_prob)


gru_cell
=
tf.nn.rnn_cell.GRUCell(encoding_dim)


outputs,
output_states
=
tf.nn.bidirectional_dynamic_rnn(gru_cell,
gru_cell,
encoded_X,
sequence_length=X_lens,
dtype=tf.float32,
swap_memory=True)


encoded_X
=
tf.concat(2,
outputs)


with
tf.variable_scope('Q'):


Q_lens
=
tf.reduce_sum(tf.sign(tf.abs(Q)),
1)


embedded_Q
=
tf.nn.embedding_lookup(embeddings,
Q)


encoded_Q
=
tf.nn.dropout(embedded_Q,
keep_prob)


gru_cell
=
tf.nn.rnn_cell.GRUCell(encoding_dim)


outputs,
output_states
=
tf.nn.bidirectional_dynamic_rnn(gru_cell,
gru_cell,
encoded_Q,
sequence_length=Q_lens,
dtype=tf.float32,
swap_memory=True)


encoded_Q
=
tf.concat(2,
outputs)


 


W_q
=
tf.Variable(tf.random_normal([2*encoding_dim,
4*encoding_dim],
stddev=0.22),
dtype=tf.float32)


b_q
=
tf.Variable(tf.random_normal([2*encoding_dim,
1],
stddev=0.22),
dtype=tf.float32)


W_d
=
tf.Variable(tf.random_normal([2*encoding_dim,
6*encoding_dim],
stddev=0.22),
dtype=tf.float32)


b_d
=
tf.Variable(tf.random_normal([2*encoding_dim,
1],
stddev=0.22),
dtype=tf.float32)


g_q
=
tf.Variable(tf.random_normal([10*encoding_dim,
2*encoding_dim],
stddev=0.22),
dtype=tf.float32)


g_d
=
tf.Variable(tf.random_normal([10*encoding_dim,
2*encoding_dim],
stddev=0.22),
dtype=tf.float32)


 


with
tf.variable_scope('attend')
as
scope:


infer_gru
=
tf.nn.rnn_cell.GRUCell(4*encoding_dim)


infer_state
=
infer_gru.zero_state(batch_size,
tf.float32)


for
iter_step
in
range(8):


if
iter_step
>
0:


scope.reuse_variables()


 


_,
q_glimpse
=
glimpse(W_q,
b_q,
encoded_Q,
infer_state)


d_attention,
d_glimpse
=
glimpse(W_d,
b_d,
encoded_X,
tf.concat_v2([infer_state,
q_glimpse],
1))


 


gate_concat
=
tf.concat_v2([infer_state,
q_glimpse,
d_glimpse,
q_glimpse
*
d_glimpse],
1)


 


r_d
=
tf.sigmoid(tf.matmul(gate_concat,
g_d))


r_d
=
tf.nn.dropout(r_d,
keep_prob)


r_q
=
tf.sigmoid(tf.matmul(gate_concat,
g_q))


r_q
=
tf.nn.dropout(r_q,
keep_prob)


 


combined_gated_glimpse
=
tf.concat_v2([r_q
*
q_glimpse,
r_d
*
d_glimpse],
1)


_,
infer_state
=
infer_gru(combined_gated_glimpse,
infer_state)


 


return
tf.to_float(tf.sign(tf.abs(X)))
*
d_attention


 


def
train_neural_attention():


X_attentions
=
neural_attention()


loss
=
-tf.reduce_mean(tf.log(tf.reduce_sum(tf.to_float(tf.equal(tf.expand_dims(A,
-1),
X))
*
X_attentions,
1)
+
tf.constant(0.00001)))


 


optimizer
=
tf.train.AdamOptimizer(learning_rate=0.001)


grads_and_vars
=
optimizer.compute_gradients(loss)


capped_grads_and_vars
=
[(tf.clip_by_norm(g,
5),
v)
for
g,v
in
grads_and_vars]


train_op
=
optimizer.apply_gradients(capped_grads_and_vars)


 


saver
=
tf.train.Saver()


with
tf.Session()
as
sess:


sess.run(tf.global_variables_initializer())


 


#
 writer = tf.summary.FileWriter()


#
 恢复前一次训练


ckpt
=
tf.train.get_checkpoint_state('.')


if
ckpt
!=
None:


print(ckpt.model_checkpoint_path)


saver.restore(sess,
ckpt.model_checkpoint_path)


else:


print("没找到模型")


 


for
step
in
range(20000):


train_x,
train_q,
train_a
=
get_next_batch()


loss_,
_
=
sess.run([loss,
train_op],
feed_dict={X:train_x,
Q:train_q,
A:train_a,
keep_prob:0.7})


print(loss_)


 


#
 保存模型并计算准确率


if
step
%
1000
==
0:


path
=
saver.save(sess,
'machine_reading.model',
global_step=step)


print(path)


 


test_x,
test_q,
test_a
=
get_test_batch()


test_x,
test_q,
test_a
=
np.array(test_x[:batch_size]),
np.array(test_q[:batch_size]),
np.array(test_a[:batch_size])


attentions
=
sess.run(X_attentions,
feed_dict={X:test_x,
Q:test_q,
keep_prob:1.})


correct_count
=
0


for
x
in
range(test_x.shape[0]):


probs
=
defaultdict(int)


for
idx,
word
in
enumerate(test_x[x,:]):


probs[word]
+=
attentions[x,
idx]


guess
=
max(probs,
key=probs.get)


if
guess
==
test_a[x]:


correct_count
+=
1


print(correct_count
/
test_x.shape[0])


 


train_neural_attention()

我只想说，这个东西比我水平高！至少在速度上能甩我十条大街。
