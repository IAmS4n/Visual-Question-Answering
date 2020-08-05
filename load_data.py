import h5py
import numpy as np
import json
import random
import datetime
import operator
import filtering

Filter = filtering.Filter()

def make_tabel_question_type(a_file):
	question_type_set = set()
	for ans_list in a_file['annotations']:
		question_type_set.add(ans_list["question_type"])
	qt2id = {qt: id for id, qt in enumerate(question_type_set)}
	return qt2id

def select_top_ans(a_path, top_ans_select_number=3000):
	#select top frequnecy ans
	a_file = json.loads(open(a_path).read())

	ans_count = {}
	for ans_list in a_file['annotations']:
		for ans_item in ans_list["answers"]:
			ans = Filter.filter_ans(ans_item["answer"])
			if ans not in ans_count:
				ans_count[ans] = 0
			else:
				ans_count[ans] += 1

	sorted_ans_tmp = sorted(ans_count.items(), key=operator.itemgetter(1))
	sorted_ans_tops_tmp = sorted_ans_tmp[-1*top_ans_select_number:]
	sorted_ans_tops = list(zip(*sorted_ans_tops_tmp))[0]

	top_ans = {ans: ans_id for ans_id, ans in enumerate(sorted_ans_tops)}
	id2ans = {ans_id: ans for ans_id, ans in enumerate(sorted_ans_tops)}
	return top_ans, id2ans


class Dataset:
	def __init__(self, general_word2vec_fd, top_ans, istrain=True, data_path="../data/", part_size=20000, load_fc7=False):
		self.istrain = istrain
		self.top_ans = top_ans
		self.load_fc7 = load_fc7
		self.top_ans_select_number = len(top_ans)

		if istrain:
			tOv = "train"
		else:
			tOv = "val"

		ordering_path = data_path + "img_ordering.json"
		q_mc_path = data_path + "MultipleChoice_mscoco_%s2014_questions.json" % tOv
		q_oe_path = data_path + "OpenEnded_mscoco_%s2014_questions.json" % tOv
		a_path = data_path + "mscoco_%s2014_annotations.json" % tOv
		img_pool5_path = data_path + "data_img_pool5.h5"
		img_fc7_path = data_path + "data_img_fc7.h5"


		self.img_pool5_file = h5py.File(img_pool5_path, 'r').get("images_%s" % tOv.replace("val", "test"))
		self.image_number = len(self.img_pool5_file)
		print("Feature number =", self.image_number)

		if self.load_fc7:
			self.img_fc7_file = h5py.File(img_fc7_path, 'r').get("images_%s" % tOv.replace("val", "test"))
			assert len(self.img_fc7_file) == self.image_number

		self.general_word2vec_fd = general_word2vec_fd
		self.pad_vectror = np.zeros(300).astype(np.float)

		################################################################
		print("Reading ordering file [%s]" % ordering_path)
		ordering_file = json.loads(open(ordering_path).read())["img_%s_ordering" % tOv]
		assert len(ordering_file) == self.image_number

		self.h5id2imgid = {}
		imgid2h5id = {}
		prefix_name = "%s2014" % tOv
		prefix_name = prefix_name + "/COCO_" + prefix_name + "_"
		for h5id, img_filepath in enumerate(ordering_file):
			assert prefix_name in img_filepath
			assert img_filepath[-4:] == ".jpg"
			imgid = int(img_filepath.replace(prefix_name, "")[:-4])
			self.h5id2imgid[h5id] = imgid
			imgid2h5id[imgid] = h5id

		################################################################
		print("Reading questions file [%s]" % q_oe_path)
		q_oe_file = json.loads(open(q_oe_path).read())
		all_questions = q_oe_file["questions"]

		if self.istrain:
			print("Reading questions file [%s]" % q_mc_path)
			q_mc_file = json.loads(open(q_mc_path).read())
			all_questions += q_mc_file["questions"]

		self.imgid2q = {}
		self.word_max = 0
		for q in all_questions:
			imgid = int(q["image_id"])
			qid = int(q["question_id"])
			qs = q["question"]

			if imgid not in imgid2h5id:
				continue

			if imgid not in self.imgid2q:
				self.imgid2q[imgid] = {}

			assert (qid not in self.imgid2q[imgid]) or (np.array_equal(qs, self.imgid2q[imgid][qid]["question"]))
			if qid in self.imgid2q[imgid]:
				continue
			self.word_max = max(self.word_max, len(self._s2m(qs)))
			self.imgid2q[imgid][qid] = {"question": qs}

		print("Maximum Len", self.word_max)
		# question_type_models = set()
		# ans_conf_models = set()
		################################################################
		mised_fetured_image = 0
		print("Reading answers file [%s]" % a_path)
		a_file = json.loads(open(a_path).read())
		self.qt2id = make_tabel_question_type(a_file)
		print("Question type num", len(self.qt2id))
		for ans_list in a_file['annotations']:
			question_type = ans_list["question_type"]
			qtid = self.qt2id[question_type]

			# question_type_models.add(question_type)

			imgid = int(ans_list["image_id"])
			qid = int(ans_list["question_id"])

			if imgid not in self.imgid2q:
				mised_fetured_image += 1
				continue

			assert qid in self.imgid2q[imgid]
			assert "answer" not in self.imgid2q[imgid][qid]
			self.imgid2q[imgid][qid]["qt"] = qtid
			self.imgid2q[imgid][qid]["answers"] = []
			for ans_item in ans_list["answers"]:
				ans = Filter.filter_ans(ans_item["answer"])
				ans_conf = ans_item["answer_confidence"]
				# ans_conf_models.add(ans_conf)

				ansid = self.a2id(ans)
				if ansid < 0:
					continue
				self.imgid2q[imgid][qid]["answers"].append(ansid)

		print("number of missed images", mised_fetured_image)

		# print(question_type_models)
		# print(ans_conf_models)

		mised_validans_num = 0
		all_question_num = 0
		for imgid in self.imgid2q:
			for qid in self.imgid2q[imgid]:
				assert "answers" in self.imgid2q[imgid][qid]
				ans_is_missed = False
				if len(self.imgid2q[imgid][qid]["answers"]) == 0:
					ans_is_missed = True
				if not self.istrain:
					is_any_valid_ans = 0
					for ans in self.imgid2q[imgid][qid]["answers"]:
						if ans >= 0:
							is_any_valid_ans += 1
					if is_any_valid_ans < 3:
						ans_is_missed = True
				if ans_is_missed:
					mised_validans_num += 1
				all_question_num += 1
		print("missed valid ans : %d / %d" % (mised_validans_num, all_question_num))

		self.DATA = []
		self.data_pointer = 0
		self.batch_pointer = 0
		self.epoch_number = 0
		self.batch_number = 0

		self.part_size = min(part_size, self.image_number)
		if self.part_size == self.image_number:
			print("[ALL DATA LOADED]")
		self.get_data_part()
		print("Part Size", self.part_size)

	def a2id(self, ans):
		if ans in self.top_ans:
			return self.top_ans[ans]
		return -1

	# def w2v(self, str):
	# 	if str in self.general_word2vec_fd:
	# 		return self.general_word2vec_fd[str]
	# 	return self.unknow_vectror


	def _s2m(self, sentence):
		listofvectors = []
		for w in Filter.filter_qus_and_split(sentence):
			if w in self.general_word2vec_fd:
				listofvectors.append(self.general_word2vec_fd[w])
		return listofvectors

	def s2m(self, sentence):
		listofvectors = self._s2m(sentence)
		real_len = len(listofvectors)
		rem = self.word_max-len(listofvectors)

		listofvectors = listofvectors + [self.pad_vectror]*rem
		# for _ in range(rem):
		# 	listofvectors.append(self.pad_vectror)

		# return np.stack(listofvectors[:self.word_max], axis=0), real_len
		return np.stack(listofvectors, axis=0), real_len

	def get_data_part(self):
		# if (self.batch_pointer + batch_size) < len(self.DATA):
		# 	return

		start_time = datetime.datetime.now()

		rem = len(self.DATA)-self.batch_pointer
		self.DATA = self.DATA[-1*rem:]

		next_data_pointer = (self.data_pointer + self.part_size) % self.image_number
		if next_data_pointer > self.data_pointer:
			data_imgpool5 = self.img_pool5_file[self.data_pointer:next_data_pointer]
			if self.load_fc7:
				data_imgfc7 = self.img_fc7_file[self.data_pointer:next_data_pointer]
			data_imgh5_id = list(range(self.data_pointer, next_data_pointer))
		else:
			data_imgpool5 = np.concatenate((self.img_pool5_file[:next_data_pointer], self.img_pool5_file[self.data_pointer:]), axis=0)
			if self.load_fc7:
				data_imgfc7 = np.concatenate((self.img_fc7_file[:next_data_pointer], self.img_fc7_file[self.data_pointer:]), axis=0)
			data_imgh5_id = list(range(0, next_data_pointer)) + list(range(self.data_pointer, self.image_number))
			self.epoch_number += 1
			print("New Epoch [%d]" % self.epoch_number)
		self.data_pointer = next_data_pointer
		self.batch_pointer = 0

		if not self.load_fc7:
			data_imgfc7 = [np.zeros(())] * len(data_imgpool5)

		data_img_id = map(self.h5id2imgid.get, data_imgh5_id)
		for img_id, img, fc in zip(data_img_id, data_imgpool5, data_imgfc7):
			reshaped_img = np.transpose(img.reshape((img.shape[0], img.shape[1]*img.shape[2])))
			#print(fc.shape) #(4096,)


			for q_id in self.imgid2q[img_id]:
				q_sentence = self.imgid2q[img_id][q_id]["question"]
				q_matrix, real_len = self.s2m(q_sentence)

				label = np.zeros([self.top_ans_select_number]).astype(np.float)

				if not self.istrain:
					each_part_prob = 0.1
				else:
					number_valid_ans = len(self.imgid2q[img_id][q_id]["answers"])
					if number_valid_ans == 0:
						continue
					each_part_prob = 1.0/float(number_valid_ans)
				for ansid in self.imgid2q[img_id][q_id]["answers"]:
					assert ansid >= 0
					label[ansid] += each_part_prob

				self.DATA.append([reshaped_img, q_matrix, label, self.imgid2q[img_id][q_id]["qt"], real_len, fc, q_id])

		if self.istrain:
			random.shuffle(self.DATA)
		end_time = datetime.datetime.now()
		print("DATA Part Reload, Time :", end_time - start_time, "Sample number :", len(self.DATA))

	def get_batch(self, batch_size):
		if self.part_size != self.image_number:
			while (self.batch_pointer + batch_size) >= len(self.DATA):
				self.get_data_part()
		self.batch_number += 1
		next_batch_pointer = (self.batch_pointer + batch_size) % len(self.DATA)

		if next_batch_pointer <= self.batch_pointer:
			next_batch = self.DATA[self.batch_pointer:] + self.DATA[:next_batch_pointer]
			random.shuffle(self.DATA)
			self.epoch_number += 1
			print("New Epoch [%d]" % self.epoch_number)
		else:
			next_batch = self.DATA[self.batch_pointer:next_batch_pointer]
		self.batch_pointer = next_batch_pointer
		return next_batch
