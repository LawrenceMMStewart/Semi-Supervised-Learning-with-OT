#tests for sinkhorn.py

import numpy as np
import tensorflow as tf
from ot.sinkhorn import *

#W(a,a) = 0
def test_sinkhorn_identity():

	x=tf.constant([[3.2,1.0,1.0],
		[1.0,1.4,1.4]],dtype =tf.float32)

	p1 = sinkhorn(x,x).numpy()
	p2 = sinkhorn(x,x).numpy()

	assert abs(p1-0)<1e-8
	assert abs(p2-0)<1e-8

#symmetry W(a,b)=W(b,a)

def test_sinkhorn_symmetry():
	x=tf.constant([[3.2,1.0,1.0],
	[1.0,1.4,1.4]],dtype =tf.float32)
	y = tf.constant([[8.9,12.0,15.0],
		[11.0,12.7,13.4],
		[19.0,13.0,14.4],
		[21.0,5.0,14.2]])
	s1 = sinkhorn(x,y).numpy()
	s2 = sinkhorn(y,x).numpy()
	print("s1",s1)
	print("s2",s2)
	assert abs(s1-s2)<1e-8


def test_sinkhorn1d_identity():
	x = tf.constant([[3.2],[7.8]],dtype=tf.float32)
	s = sinkhorn(x,x).numpy()
	assert s<1e-8

def test_sinkhorn1d_symmetry():
	x = tf.constant([[3.2],[7.8]],dtype=tf.float32)
	y = tf.constant([[4.2],[9.8]],dtype=tf.float32)
	s = sinkhorn(x,y).numpy()
	s2 = sinkhorn(y,x).numpy()
	assert abs(s-s2)<1e-8