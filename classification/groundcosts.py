import tensorflow as tf
import spacy
import numpy as np
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt 
import pickle
import argparse


parser = argparse.ArgumentParser('parameters')
parser.add_argument('--c10', action='store_true', help='run cifar 10')
parser.add_argument('--c100', action='store_true', help='run cifar 100')
args= parser.parse_args()


if args.c10:
	def generate_cifar10_corpus():
		labels = ""
		labels += "airplane"
		labels += " automobile"
		labels += " bird"
		labels += " cat"
		labels += " deer"
		labels += " dog"
		labels += " frog"
		labels += " horse"
		labels += " ship"
		labels += " truck"
		return labels

	def generate_cifar10_labels():
		"""
		comma seperated labels
		for cifar 10
		"""
		labels = []
		labels.append("airplane")
		labels.append("automobile")
		labels.append("bird")
		labels.append("cat")
		labels.append("deer")
		labels.append("dog")
		labels.append("frog")
		labels.append("horse")
		labels.append("ship")
		labels.append("truck")
		return labels

	#load spacy model 
	nlp = spacy.load('en_core_web_sm')
	# process the cifar names using the model
	doc = nlp(generate_cifar10_corpus())

	veclist = []
	for i in range(len(doc)):
		veclist.append(doc[i].vector.reshape(1,-1))

	X = np.concatenate(veclist,axis=0)

	C = pairwise_distances(X)
	Cnorm = C/np.median(C)

	np.save("./groundmetrics/nm-cifar10.npy",Cnorm)

	sns.heatmap(Cnorm,xticklabels=generate_cifar10_labels(),
		yticklabels=generate_cifar10_labels(),cmap="YlGnBu")
	plt.show()


if args.c100:

	coarse_labels = [
	'apple', # id 0
	'aquarium_fish',
	'baby',
	'bear',
	'beaver',
	'bed',
	'bee',
	'beetle',
	'bicycle',
	'bottle',
	'bowl',
	'boy',
	'bridge',
	'bus',
	'butterfly',
	'camel',
	'can',
	'castle',
	'caterpillar',
	'cattle',
	'chair',
	'chimpanzee',
	'clock',
	'cloud',
	'cockroach',
	'couch',
	'crab',
	'crocodile',
	'cup',
	'dinosaur',
	'dolphin',
	'elephant',
	'flatfish',
	'forest',
	'fox',
	'girl',
	'hamster',
	'house',
	'kangaroo',
	'computer_keyboard',
	'lamp',
	'lawn_mower',
	'leopard',
	'lion',
	'lizard',
	'lobster',
	'man',
	'maple_tree',
	'motorcycle',
	'mountain',
	'mouse',
	'mushroom',
	'oak_tree',
	'orange',
	'orchid',
	'otter',
	'palm_tree',
	'pear',
	'pickup_truck',
	'pine_tree',
	'plain',
	'plate',
	'poppy',
	'porcupine',
	'possum',
	'rabbit',
	'raccoon',
	'ray',
	'road',
	'rocket',
	'rose',
	'sea',
	'seal',
	'shark',
	'shrew',
	'skunk',
	'skyscraper',
	'snail',
	'snake',
	'spider',
	'squirrel',
	'streetcar',
	'sunflower',
	'sweet_pepper',
	'table',
	'tank',
	'telephone',
	'television',
	'tiger',
	'tractor',
	'train',
	'trout',
	'tulip',
	'turtle',
	'wardrobe',
	'whale',
	'willow_tree',
	'wolf',
	'woman',
	'worm',
	]


	def generate_cifar100_corpus():
		labels = coarse_labels[0]
		for i in range(1,len(coarse_labels)):
			labels+=" "+coarse_labels[i]
		return labels

	def generate_cifar100_labels():
		return coarse_labels




	#load spacy model 
	nlp = spacy.load('en_core_web_sm')
	# process the cifar names using the model
	doc = nlp(generate_cifar100_corpus())

	veclist = []
	for i in range(len(doc)):
		veclist.append(doc[i].vector.reshape(1,-1))

	X = np.concatenate(veclist,axis=0)

	C = pairwise_distances(X)
	Cnorm = C/np.median(C)

	np.save("./groundmetrics/nm-cifar100.npy",Cnorm)

	sns.heatmap(Cnorm,xticklabels=generate_cifar100_labels(),
		yticklabels=generate_cifar100_labels(),cmap="YlGnBu")
	plt.xticks(fontsize=4, rotation=90)
	plt.yticks(fontsize=4, rotation=0)
	plt.show()
