import load_data
import model
import tensorflow as tf
import gensim
import random
import numpy as np
import datetime
import json

top_ans_number = 3000
batch_size = 128
validation_size = 1000
save_path = "./save/"
data_path = "../data/"

def split_batch(packed):
	img, q, a, qt, rl, fc, qid = zip(*packed)
	return img, q, a, qt, rl, fc, qid


print("Selecting Top %d Answers" % top_ans_number)
a_path = data_path + "mscoco_train2014_annotations.json"
ans2id, id2ans = load_data.select_top_ans(a_path, top_ans_number)

print("Reading W2V file")
w2v_path = data_path + "GoogleNews-vectors-negative300.bin" #"pruned_w2v.bin"
word2vec_fd = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)

print("="*30 + " Validation " + "="*30)
Dataset_Val = load_data.Dataset(general_word2vec_fd=word2vec_fd, top_ans=ans2id, data_path=data_path, istrain=False, part_size=validation_size)

packed_valbatch = Dataset_Val.get_batch(validation_size)
batch_valimg, batch_valq, batch_vala, _, batch_valrl, batch_valfc, _ = split_batch(packed_valbatch)

print("="*30 + " Train " + "="*30)
Dataset_Train = load_data.Dataset(general_word2vec_fd=word2vec_fd, top_ans=ans2id, data_path=data_path, istrain=True, part_size=9999999)



# for tmp in Dataset_Train.get_batch(10):
# 	print(tmp[2])
# 	print(tmp[1].shape, tmp[1].dtype, tmp[1].max(), tmp[1].min(), tmp[1].mean())
# 	print(tmp[0].shape, tmp[0].dtype, tmp[0].max(), tmp[0].min(), tmp[0].mean())


def accuracy(outs, ans):
	outs = list(outs)

	# s = random.randint(0, len(ans)-11)
	# best_ans = []
	# for ans_list in ans[s:s+10]:
	# 	best_ans.append(np.argmax(ans_list))
	# print(map(id2ans.get, outs[s:s+10]))
	# print(map(id2ans.get, best_ans))

	assert len(outs) == len(ans)
	correct_ans = 0.0
	for predict, ans_list in zip(outs, ans):
		correct_ans += min(1, (ans_list[predict]*10)/3.0)
	return float(correct_ans)/float(len(outs))


sess = tf.Session()
max_len = max(Dataset_Train.word_max, Dataset_Val.word_max)
Model = model.Model(class_number=top_ans_number, max_len=max_len)

saver = tf.train.Saver()

tf.summary.scalar("loss", Model.loss)
tf.summary.scalar("Train_acc", Model.selected_ans_accuracy)
merge_summary = tf.summary.merge_all()
filewriter = tf.summary.FileWriter("./log/%s/" % datetime.datetime.now().strftime("%Y_%m_%d__%H_%M"))

if False:
	saver.restore(sess, save_path)
	print("Network loaded from %s" % save_path)
else:
	sess.run(tf.global_variables_initializer())


last_save_epoch = -1
while Dataset_Train.epoch_number < 100:
	packed_batch = Dataset_Train.get_batch(batch_size)
	batch_img, batch_q, batch_a, batch_qt, batch_rl, batch_fc, _ = split_batch(packed_batch)

	feed_dict = {Model.correct_label_dis: batch_a,
					Model.inp_img: batch_img,
	                # Model.inp_img_fc: batch_fc,
					Model.inp_question: batch_q,
	                Model.inp_question_len: batch_rl,
					# Model.correct_label_qt: batch_qt,
					Model.IsTrain: True}
	# print(sess.run(tf.shape(Model.img_content_tmp), feed_dict=feed_dict))
	# print(sess.run(tf.shape(correct_vector), feed_dict=feed_dict))
	sess.run(Model.optimizer, feed_dict=feed_dict)
	if last_save_epoch < Dataset_Train.epoch_number:
		csummary = sess.run(merge_summary, feed_dict=feed_dict)

		feed_dict = {Model.inp_img: batch_valimg,
		                # Model.inp_img_fc: batch_valfc,
		                Model.inp_question: batch_valq,
		                Model.inp_question_len: batch_valrl,
						Model.IsTrain: False}
		batch_prediction = sess.run(Model.prediction, feed_dict=feed_dict)

		validation_mesure = accuracy(batch_prediction, batch_vala)
		print(validation_mesure)
		validation_summary = tf.Summary()
		validation_summary.value.add(tag='Validation Accuracy', simple_value=validation_mesure)

		filewriter.add_summary(csummary, Dataset_Train.batch_number)
		filewriter.add_summary(validation_summary, Dataset_Train.batch_number)
		# print("Loss = %f , Train_acc = %f  , Validation_mesure = %f" % (loss, train_acc, test_mesure))

		saver.save(sess=sess, save_path=save_path)
		last_save_epoch = Dataset_Train.epoch_number
		if Dataset_Train.epoch_number % 7 == 0:
			Dataset_Test = load_data.Dataset(general_word2vec_fd=word2vec_fd, top_ans=ans2id, data_path=data_path, istrain=False, part_size=99999999)
			results = []
			item_remind = len(Dataset_Test.DATA)
			while item_remind > 0:
				packed_batch = Dataset_Test.get_batch(min(batch_size, item_remind))
				item_remind -= batch_size
				batch_testimg, batch_testq, _, _, batch_testrl, _, batch_qid = split_batch(packed_batch)
				feed_dict = {Model.inp_img: batch_testimg,
					                Model.inp_question: batch_testq,
					                Model.inp_question_len: batch_testrl,
									Model.IsTrain: False}
				batch_prediction = sess.run(Model.prediction, feed_dict=feed_dict)
				for qid, predict in zip(batch_qid, batch_prediction):
					results.append({"question_id": qid, "answer": id2ans[predict]})

			with open("results_%d_%s.json" % (Dataset_Train.epoch_number, validation_mesure), 'w') as f:
				json.dump(results, f)
