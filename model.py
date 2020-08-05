import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class Model:
	def __init__(self, class_number, max_len):
		self.inp_img_fn = 512
		self.inp_img_fn_manipulate = self.inp_img_fn + 2
		self.class_number = class_number
		self.attention_number = 3
		self.qt_class_number = 65

		self.lstm_hidden_size = 1024
		self.filter_num1 = 512 # in attention
		self.fc1_num = 1024 # in classifier

		self.inp_question = tf.placeholder(tf.float32, [None, max_len, 300], name="q_inp")
		self.inp_question_len = tf.placeholder(tf.int64, [None, ], name="q_inp_len")
		self.inp_img = tf.placeholder(tf.float32, [None, 49, self.inp_img_fn], name="q_inp")
		self.inp_img_fc = tf.placeholder(tf.float32, [None, 4096], name="q_inp_fc")
		self.correct_label_dis = tf.placeholder(tf.float32, [None, class_number], name="correct_CLASS_label")
		self.correct_label_qt = tf.placeholder(tf.int64, [None, ], name="correct_QT_label")
		self.IsTrain = tf.placeholder(tf.bool, shape=())

		inp_img_norm = self.positional(tf.nn.l2_normalize(self.inp_img, -1))

		# rnn_last_output = self.lstm1(tf.tanh(self.inp_question))
		rnn_last_output = self.lstm2(tf.tanh(self.inp_question), self.inp_question_len)

		o_qt = tf.layers.dense(
				inputs=rnn_last_output,
				units=self.qt_class_number,
				activation=None
		)
		loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.correct_label_qt, logits=o_qt))

		attentions = self.attetntion2(image=inp_img_norm, question_features=rnn_last_output)
		img_atend = self.apply_attention(image=inp_img_norm, attentions=attentions) #batchsize,512,2
		img_atend_flat = tf.reshape(img_atend, [-1, self.attention_number*self.inp_img_fn_manipulate])#batchsize,1024

		content = tf.concat([img_atend_flat, rnn_last_output], axis=1)# batchsize,1024+1024
		# content = tf.concat([), rnn_last_output], axis=1)
		img_flat = tf.reshape(inp_img_norm, [-1, self.inp_img_fn_manipulate*7*7])
		# content = tf.concat([self.inp_img_fc, img_atend_flat, rnn_last_output], axis=1)

		d_f1 = tf.layers.dropout(inputs=content, rate=0.5, training=self.IsTrain)
		o_f1 = tf.layers.dense(
				inputs=d_f1,
				units=self.fc1_num,
				activation=tf.nn.relu,
				bias_initializer=tf.constant_initializer(0.1)
		)

		d_f2 = tf.layers.dropout(inputs=o_f1, rate=0.5, training=self.IsTrain)
		o_f2 = tf.layers.dense(
				inputs=d_f2,
				units=self.class_number,
				activation=None)

		loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.correct_label_dis, logits=o_f2))
		# self.loss = loss1 + loss2
		self.loss = loss2
		step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(learning_rate=1e-3,
													global_step=step,
													decay_steps=50000,
													decay_rate=0.5)
		self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

		self.correct_label_int = tf.argmax(self.correct_label_dis, 1)
		self.prediction = tf.argmax(o_f2, 1)
		correct_pred = tf.equal(self.prediction, self.correct_label_int)
		self.selected_ans_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	def attetntion1(self, image, question_features):

		d_question = tf.layers.dropout(inputs=question_features, rate=0.5, training=self.IsTrain)
		o_rnn = tf.layers.dense(
				inputs=d_question,
				units=self.filter_num1,
				activation=tf.nn.relu,
				bias_initializer=tf.constant_initializer(0.1)
		)

		d_conv1 = tf.layers.dropout(inputs=image, rate=0.5, training=self.IsTrain)
		o_conv1 = tf.layers.conv1d(
					inputs=d_conv1,
					filters=self.filter_num1,
					kernel_size=1,
					strides=1,
					padding="same",
					activation=tf.nn.relu,
					bias_initializer=tf.constant_initializer(0.1)
		)# batchsize,49,512
		o_rnn_tile_tmp = tf.reshape(o_rnn, [-1, 1, self.filter_num1])
		o_rnn_tile = tf.tile(o_rnn_tile_tmp, [1, 49, 1])
		merged = tf.concat([o_conv1, o_rnn_tile], axis=-1)
		d_conv2 = tf.layers.dropout(inputs=merged, rate=0.5, training=self.IsTrain)
		o_conv2 = tf.layers.conv1d(
					inputs=d_conv2,
					filters=self.attention_number,
					kernel_size=1,
					strides=1,
					padding="same",
					activation=None)# batchsize,49,2

		attention = tf.nn.softmax(o_conv2, 1)
		return attention


	def attetntion2(self, image, question_features):
		o_rnn_tile_tmp = tf.reshape(question_features, [-1, 1, self.lstm_hidden_size])
		o_rnn_tile = tf.tile(o_rnn_tile_tmp, [1, 49, 1])
		merged = tf.concat([image, o_rnn_tile], axis=-1)

		d_conv1 = tf.layers.dropout(inputs=merged, rate=0.5, training=self.IsTrain)
		o_conv1 = tf.layers.conv1d(
					inputs=d_conv1,
					filters=self.filter_num1,
					kernel_size=1,
					strides=1,
					padding="same",
					activation=tf.nn.relu,
					bias_initializer=tf.constant_initializer(0.1)
		)# batchsize,49,512

		d_conv2 = tf.layers.dropout(inputs=o_conv1, rate=0.5, training=self.IsTrain)
		o_conv2 = tf.layers.conv1d(
					inputs=d_conv2,
					filters=self.attention_number,
					kernel_size=1,
					strides=1,
					padding="same",
					activation=None)# batchsize,49,2

		attention = tf.nn.softmax(o_conv2, 1)
		return attention


	def lstm1(self, inp):

		lstm_cell = rnn.BasicLSTMCell(self.lstm_hidden_size, forget_bias=1.0)
		inp_q_list = tf.unstack(inp, axis=1)

		rnn_outputs, _ = rnn.static_rnn(lstm_cell, inp_q_list, dtype=tf.float32)
		return rnn_outputs[-1]


	def lstm2(self, inp, lens):
		# lstm_dropout = rnn.DropoutWrapper(lstm_layer, input_keep_prob=dropout_rate)  # apply dropout on lstm inputs
		batch_size = tf.shape(inp)[0]
		lstm_cell = rnn.BasicLSTMCell(self.lstm_hidden_size, forget_bias=1.0)
		init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
		rnn_outputs, rnn_states = tf.nn.dynamic_rnn(lstm_cell, inp, initial_state=init_state, dtype=tf.float32, sequence_length=lens)
		return rnn_states.h

	def apply_attention(self, image, attentions):
		# image : batchsize,49,512   attention :batchsize,49,2
		inner_product_func = lambda xy: tf.tensordot(xy[0], xy[1], axes=[[0], [0]])
		return tf.map_fn(inner_product_func, elems=(image, attentions), dtype=tf.float32)

	def positional(self, image):
		pos = []
		for x in range(7):
			for y in range(7):
				pos.append(x)
				pos.append(y)
		nav = tf.constant(np.array(pos).reshape(49, 2), dtype=tf.float32)

		concat_func = lambda x: tf.concat([x, nav], axis=1)
		return tf.map_fn(concat_func, elems=(image), dtype=tf.float32)


