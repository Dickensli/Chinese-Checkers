import tensorflow as tf
import random
import os
from cc_agent import Agent

flags = tf.app.flags
#flags.DEFINE_string('gpu_fraction', '2/3', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_integer('mode', 1, 'three modes')
flags.DEFINE_integer('layer', 6, 'three layers')
flags.DEFINE_integer('type', 0, 'three types')
FLAGS = flags.FLAGS
'''
def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction
'''
def main(_):
  
  #gpu_options = tf.GPUOptions(
  #    per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
  #os.environ["CUDA_VISIBLE_DEVICES"]="1"
  #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  with tf.Session() as sess:
    random.seed(None)
    
    agent = Agent(sess)
    if(FLAGS.layer != 4 and FLAGS.layer != 6 and FLAGS.layer !=8):
       print "Look ahead layer can only be 4,6,8"
       quit()
    if(FLAGS.type != 0 and FLAGS.type != 1 and FLAGS.type !=2):
       print "Wrong agent type. 0:normal, 1: defensive, 2: offensive"
       quit()
    
    if (FLAGS.mode == 1):
        agent.train_stage1()
    elif(FLAGS.mode == 2):
        agent.train_stage2()
    elif(FLAGS.mode == 3):
        print "Look Ahead layer: ", FLAGS.layer
        print "Agent Type: ", FLAGS.type
        agent.play(FLAGS.layer,FLAGS.type)

if __name__ == '__main__':
  tf.app.run()
