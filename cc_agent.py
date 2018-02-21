import sys
import math
import shelve
import random
import tensorflow as tf
import env
#import env_test
import env_off
import env_def
#import env_nn
import csv
import numpy as np

from math import *

class Agent():
  def __init__(self, sess):
    self.sess = sess
    self.build_nn()
    self.PHASE3_FLAG = False
    self.game_board =[[0, 0, 0, 0, 0, -1, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 1, 0, 0, 0, 0, 0]]
    self.rotated_board=[[0, 0, 0, 0, 0, -1, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 1, 0, 0, 0, 0, 0]]
    self.chessID = [(0, 8), (0, 7), (1, 8), (0, 6), (1, 7), (2, 8), (0, 5), (1, 6), (2, 7), (3, 8), 
                    (8, 0), (7, 0), (8, 1), (6, 0), (7, 1), (8, 2), (5, 0), (6, 1), (7, 2), (8, 3)]
    self.rotated_chessID = [(0, 8), (0, 7), (1, 8), (0, 6), (1, 7), (2, 8), (0, 5), (1, 6), (2, 7), (3, 8), 
                    (8, 0), (7, 0), (8, 1), (6, 0), (7, 1), (8, 2), (5, 0), (6, 1), (7, 2), (8, 3)]
    self.TrainingBoard=[[-99,-99,-99,-99,-33,-30,-35,-40,-55],
            [-99,-99,-10,-15,-20,-25,-30,-35,-40],
            [-99,-3,-5,-7,-14,-20,-25,-30,-35],
            [-99,-3,-2,-5,-7,-14,-20,-25,-30],
            [-2,-1,0,-2,-5,-7,-14,-21,-33],
            [1,6,1,-1,-4,-6,-7,-15,-99],
            [2,1,1,0,1,-2,-6,-10,-99],
            [3,6,6,0,-4,-2,-5,-99,-99],
            [2.5,3,2,3,-3,-99,-99,-99,-99]]

    self.diff = 0
    self.turn = 1
    self.old_turn = 0
    self.round = 0
    self.found_red_phase3 = False
    self.found_blue_phase3 = False
    self.blue_finished = False
    self.red_finished = False
    self.blue_end_layer = 0
    self.red_end_layer = 0
    self.guess_count = 0
    #self.blue_step = 0
    #self.red_step = 0
    self.blue_record = {"l":[],"b":[]}
    self.red_record = {"l":[],"b":[]}
    #self.red_bad = {"l":[],"b":[]}
    self.db3 = shelve.open('phase3.db')
    self.terminate = 4000
    self.gramma = 0.8
    self.BESTMOVE = []
    self.hash_key32 = []
    self.hash_key64 = []
    self.key32 = 0
    self.key64=0
    self.TRANSTABLE = {}
    self.maxDepth = 4
    self.tempDepth = 0
    self.win = [(0,8),(0,7),(1,8),(0,6),(1,7),(2,8),(0,5),(1,6),(2,7),(3,8)]  
  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=True)
  def bias_variable(self,shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=True)  
    
  def get_last(self,chessid,color):
    if(color == 2):
      min = chessid[0][0] - chessid[0][1]
      for i in range(1,10):
        if((chessid[i][0] - chessid[i][1]) < min): min = (chessid[i][0] - chessid[i][1])
      return min
    elif(color == 3):
      max = chessid[10][0] - chessid[10][1]
      for i in range(11,20):
        if((chessid[i][0] - chessid[i][1]) > max): max = (chessid[i][0] - chessid[i][1])    
      return max
      
  def is_same(self,list,win):
    count = 0
    for element1 in list :
      for element2 in win :
        if (element1 == element2): 
          count += 1
          break
  
    if(count == 10) : return True
    else : return False    
      
  def isSeparate(self,chessid):
    last_pos_red = self.get_last(chessid,3)
    last_pos_blue = self.get_last(chessid,2)
    if(last_pos_blue > last_pos_red): return True 
    else: return False    
    
  def gen_step(self,chessid,board,color):
    result = []
    stack = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    temp_movelist = []
    if(color == 2):
      for i in range(10,20):

        (x,y) = chessid[i]
        stack[0]=(x,y)
        temp=[(x,y)]
        stack_len = 1;
        while (stack_len > 0):        
            (x1,y1) = stack[stack_len-1]
            stack_len -= 1

            if(x1 < 2): 
                flag_x1 = False
                flag_x2 = True
            elif(x1 > 6):
                flag_x1 = True
                flag_x2 = False
            else:
                flag_x1 = True
                flag_x2 = True
            if(y1 < 2): 
                flag_y1 = False
                flag_y2 = True
            elif(y1 > 6):
                flag_y1 = True
                flag_y2 = False
            else:
                flag_y1 = True
                flag_y2 = True
        
            if(flag_x1 == True and flag_y1 == True  and board[x1-1][y1-1] != 0 and board[x1-2][y1-2] == 0 and ((x1-2,y1-2) not in temp)):

                stack[stack_len] = (x1-2,y1-2)
                stack_len += 1
                temp.append((x1-2,y1-2))
            if(flag_x2 == True and flag_y2 == True and board[x1+1][y1+1] != 0 and board[x1+2][y1+2] == 0 and   ((x1+2,y1+2) not in temp)):

                stack[stack_len] = (x1+2,y1+2)   
                stack_len += 1
          
                temp.append((x1+2,y1+2))
            if(flag_x2 == True and board[x1+1][y1] != 0 and board[x1+2][y1] == 0 and  ((x1+2,y1) not in temp)):

                stack[stack_len] = (x1+2,y1)
                stack_len += 1
                temp.append((x1+2,y1))
            if(flag_y1 == True  and board[x1][y1-1] != 0 and board[x1][y1-2] == 0 and ((x1,y1-2) not in temp)):

                stack[stack_len] = (x1,y1-2)    
                stack_len += 1            
                temp.append((x1,y1-2))
            if(flag_y2 == True and board[x1][y1+1] != 0 and board[x1][y1+2] == 0 and ((x1,y1+2) not in temp)):
                stack[stack_len] = (x1,y1+2)            
                stack_len += 1          
                temp.append((x1,y1+2))
            if(flag_x1 == True and board[x1-1][y1] != 0 and board[x1-2][y1] == 0 and ((x1-2,y1) not in temp)):

                stack[stack_len] = (x1-2,y1)  
                stack_len += 1          
                temp.append((x1-2,y1))
  
        temp.remove((x,y))

        if(x>=1 and x <=9 and y >=1 and y <=9 and board[x-1][y-1] == 0):
                temp.append((x-1,y-1))  
        if(x >=-1 and x <=7 and y >=-1 and y <=7 and board[x+1][y+1] == 0):
                temp.append((x+1,y+1))
        if(x >=0 and x <=8 and y >=-1 and y <=7 and board[x][y+1] == 0):
                temp.append((x,y+1))
        if(x >=1 and x <=9 and y >=0 and y <=8 and board[x-1][y] == 0):
                temp.append((x-1,y))
      
        temp_movelist.append(temp)
        
      for i in range(10):  
        old = chessid[i+10]
        for j in range(len(temp_movelist[i])):
          (x,y) = temp_movelist[i][j]
          
          if((y-x) >= (old[1]-old[0]) and self.TrainingBoard[8-x][8-y] != -99):
            result.append({"toPos": (x,y), "fromPos": old})
            
            
    elif(color == 1):
      for i in range(10):
        (x,y) = chessid[i]

        stack[0]=(x,y)

        temp=[(x,y)]
        stack_len = 1


        while (stack_len > 0):
          (x1,y1) = stack[stack_len-1]
          stack_len -= 1          
          if(x1 < 2): 
            flag_x1 = False
            flag_x2 = True
          elif(x1 > 6):
            flag_x1 = True
            flag_x2 = False
          else:
            flag_x1 = True
            flag_x2 = True
          if(y1 < 2): 
            flag_y1 = False
            flag_y2 = True
          elif(y1 > 6):
            flag_y1 = True
            flag_y2 = False
          else:
            flag_y1 = True
            flag_y2 = True
          if(flag_x1 == True and flag_y1 == True and board[x1-1][y1-1] != 0 and board[x1-2][y1-2] == 0 and ((x1-2,y1-2) not in temp)):
            stack[stack_len]=(x1-2,y1-2)
            stack_len += 1
            temp.append((x1-2,y1-2))

          if(flag_x2 == True and flag_y2 == True and board[x1+1][y1+1] != 0 and board[x1+2][y1+2] == 0 and   ((x1+2,y1+2) not in temp)):

            stack[stack_len]=(x1+2,y1+2)
            stack_len += 1
            temp.append((x1+2,y1+2))

          if(flag_x2 == True and board[x1+1][y1] != 0 and board[x1+2][y1] == 0 and  ((x1+2,y1) not in temp)):

            stack[stack_len]=(x1+2,y1)
            stack_len += 1            
            temp.append((x1+2,y1))

          if(flag_y1 == True  and board[x1][y1-1] != 0 and board[x1][y1-2] == 0 and ((x1,y1-2) not in temp)):

            stack[stack_len]=(x1,y1-2)
            stack_len += 1            
            temp.append((x1,y1-2))


          if(flag_y2 == True and board[x1][y1+1] != 0 and board[x1][y1+2] == 0 and ((x1,y1+2) not in temp)):
              stack[stack_len]=(x1,y1+2)
              stack_len += 1
              temp.append((x1,y1+2))

          if(flag_x1 == True and board[x1-1][y1] != 0 and board[x1-2][y1] == 0 and ((x1-2,y1) not in temp)):
              stack[stack_len]=(x1-2,y1)
              stack_len += 1
              temp.append((x1-2,y1))

  
        temp.remove((x,y))
  
        if(x >=1 and x <=9 and y>=1 and y<=9 and board[x-1][y-1] == 0):
            temp.append((x-1,y-1)) 

        if(x>=-1 and x <=7 and y >=-1 and y <=7 and board[x+1][y+1] == 0):
            temp.append((x+1,y+1))

        if(x >=-1 and x<=7 and y >=0 and y <=8 and board[x+1][y] == 0):
            temp.append((x+1,y))

        if(x >=0 and x <=8 and y >=1 and y <=9 and board[x][y-1] == 0):
            temp.append((x,y-1))

        
        temp_movelist.append(temp) 
        
      for i in range(10):  
        old = chessid[i]
        for j in range(len(temp_movelist[i])):
            (x,y) = temp_movelist[i][j]
            
            if((y-x) <= (old[1]-old[0]) and self.TrainingBoard[x][y] != -99):
                result.append({"toPos": (x,y), "fromPos": old})
 
    return result
  


  def build_nn(self):
    self.input_board = tf.placeholder(tf.float32, shape = [None, 81])
    self.label = tf.placeholder(tf.float32, shape = [None, 1])
    #self.output = tf.placeholder(tf.float32, shape = [None, 1])
    self.keep_prob = tf.placeholder(tf.float32)
    self.discount = tf.placeholder(tf.float32)
    self.scale = tf.placeholder(tf.float32)
    #self.sm = tf.placeholder(tf.float32, shape = [None, 1])
    self.col_index = tf.placeholder(tf.int32)
    #self.x_image = tf.reshape(self.input, [-1,9,9,1])

    #x_print1 = tf.Print(h_pool2[0],[h_pool2[0]],message="aaaa:")
    #x_print = tf.Print(x_print1[0],[x_print1[0]],message="aaaa:")
    self.W_fc1 = self.weight_variable([81 , 729])
    self.b_fc1 = self.bias_variable([729])
    self.h_pool1_flat = tf.reshape(self.input_board, [-1, 9*9])
    self.h_nor1 = tf.nn.l2_normalize(tf.matmul(self.h_pool1_flat, self.W_fc1) + self.b_fc1,1)
    self.h_fc1 = tf.nn.relu(self.h_nor1)
    
    
    self.W_fc2 = self.weight_variable([729, 729])
    self.b_fc2 = self.bias_variable([729])
    #self.h_pool3_flat = tf.reshape(self.h_fc1, [-1, 9*9])
    self.h_nor2 = tf.nn.l2_normalize(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2,1)
    self.h_fc2 = tf.nn.relu(self.h_nor2)
    
    '''
    self.W_fc3 = self.weight_variable([1024 , 512])
    self.b_fc3 = self.bias_variable([512])
    #self.h_pool2_flat = tf.reshape(self.input_board, [-1, 9*9])
    self.h_fc3 = tf.nn.relu(tf.matmul(self.h_nor2, self.W_fc3) + self.b_fc3)
    self.h_nor3 = tf.nn.l2_normalize(self.h_fc3,1)
    #self.x_print1 = tf.Print(self.h_nor3[12][511],[self.h_nor3[12][511]],message="aaaa:")
    '''
    self.h_fc1_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)
    self.W_fc4 = self.weight_variable([729, 1])
    self.b_fc4 = self.bias_variable([1])
    
    #y_temp=tf.matmul(h_fc1_drop, W_fc2)
    self.output = tf.matmul(self.h_fc1_drop, self.W_fc4) + self.b_fc4
    #self.x_print1 = tf.Print(self.temp,[self.temp],message="aaaa:")
    self.predict=tf.nn.softmax(self.output, dim=0)
    #self.x_print2 = tf.Print(self.predict,[self.predict],message="cccc:")
    #self.x_print1 = tf.Print(self.output[10][0],[self.output[10][0]],message="aaaa:")
    self.target = tf.gather(self.predict, self.col_index)
    
    self.squared_error1 = tf.pow(self.predict - self.label,2)*0.5*self.discount*self.scale
    self.squared_error2 = tf.pow((self.target - self.label),2)*0.5 *self.discount*self.scale
    
    
    #self.temp =tf.Print(self.squared_error,[self.squared_error],message="aaaa:")
    #self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.x_print), 1))
    #x_print = tf.Print(cross_entropy,[cross_entropy],message="aaaa:")
    
    
    self.train_step1 = tf.train.AdamOptimizer(1e-4).minimize(self.squared_error1)
    self.train_step2 = tf.train.AdamOptimizer(1e-4).minimize(self.squared_error2)
    
    
    
    self.arg_max = tf.argmax(self.predict,0)
    #self.temp1 = tf.argmax(self.label,0)
    #self.x_print = tf.Print(self.temp,[self.temp],message="aaaa:")
    #self.x_print1 = tf.Print(self.temp1,[self.temp1],message="bbbb:")
    self.correct_prediction = tf.equal(self.arg_max, tf.argmax(self.label,0))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    

    
  def train_stage1(self):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('')
    if ckpt and ckpt.model_checkpoint_path:
        print "continue stage 1"
        saver.restore(self.sess,"./cc_model")
    else:
        self.sess.run(tf.global_variables_initializer())    
    
    while True:
      if(self.old_turn != self.turn):
        self.old_turn = self.turn
      ###blue
      if(self.turn % 2 == 1 ):
        #if(self.red_finished == True ): self.diff += 1
        
        ####rotated_chessID
        ######
        if(self.turn ==1):
          if(self.round%4 == 1 or self.round%4 == 0):
            new_x = 5
            new_y = 1
            old_x = 5
            old_y = 0
          else:
            new_x = 7
            new_y = 3
            old_x = 8
            old_y = 3
        else:
          reduce_dim = []
          for i in range(9):
            for j in range(9):
              reduce_dim.append(self.rotated_board[i][j])
              
          #if(self.turn >= 50): reduce_dim[80] = 1    
          env_out = env.negascout(reduce_dim)
          
          new_x = env_out[0][0]
          new_y = env_out[0][1]
          old_x = env_out[1][0]
          old_y = env_out[1][1]
        
        move_list = self.gen_step(self.chessID,self.game_board,1)
        #######train#######
        board_list = []
        label_list = []
        flag = False
        for i in range(len(move_list)):
            (tox,toy) = move_list[i]["toPos"]
            (fromx,fromy) = move_list[i]["fromPos"]
            if((tox,toy) == (8-new_x,8-new_y) and (fromx,fromy) == (8-old_x,8-old_y)):
              label_list.append([1])
              flag = True
            else: label_list.append([0]) 
            
            self.rotated_board[8-fromx][8-fromy] = 0
            self.rotated_board[8-tox][8-toy] = 1
            
            temp_board = np.asarray(self.rotated_board,dtype=np.float32) *0.5
            '''
            for m in range(9):
              for n in range(9):
                if(temp_board[m][n] != 0): temp_board[m][n] -= 1.5
            '''    
            temp_board = np.reshape(temp_board,81)
            #value = self.predict.eval(feed_dict={
            #    self.input_board: [temp_board], self.keep_prob: 1.0})
            #print temp_board
            board_list.append(temp_board)
            self.rotated_board[8-fromx][8-fromy] = 1
            self.rotated_board[8-tox][8-toy] = 0
          
        if(flag == False):
            print "ERROR!!"
            quit()

        label_list = np.asarray(label_list)
        self.train_step1.run(feed_dict={self.input_board: board_list, self.label: label_list, self.keep_prob: 0.5, self.discount: 1.0, self.scale:1.0})
        train_accuracy = self.accuracy.eval(feed_dict={self.input_board: board_list, self.label: label_list, self.keep_prob: 1.0, self.discount: 1.0, self.scale:1.0})
        self.guess_count += train_accuracy
          
        #ep = self.round *0.5 / self.terminate + 0.5
        ##random
        
        if(random.random() < 0.5 and self.turn > 10):
            ran = random.randrange(6)
            #index = random.randrange(len(move_list))
            value_list = np.asarray(self.output.eval(feed_dict={self.input_board: board_list, self.keep_prob: 1.0}))
            #print value_list
            value_list = value_list[:,0]
            index = value_list.argsort()[-6:][::-1]
            ran = min(ran,len(index)-1)
            (new_x,new_y) = (8-move_list[index[ran]]["toPos"][0],8-move_list[index[ran]]["toPos"][1])
            (old_x,old_y) = (8-move_list[index[ran]]["fromPos"][0],8-move_list[index[ran]]["fromPos"][1])
        

          ###################
        #######next move#######
        '''
        print "blue:"
        print (8-old_x,8-old_y)
        print (8-new_x,8-new_y)
        '''
        self.game_board[8-old_x][8-old_y] = 0
        self.game_board[8-new_x][8-new_y] = -1
        
        self.rotated_board[old_x][old_y] = 0
        self.rotated_board[new_x][new_y] = 1
        
        index = self.chessID.index((8-old_x,8-old_y))
        self.chessID[index] = (8-new_x,8-new_y)
        ##############
      
      #####red  
      elif(self.turn % 2 == 0 ):
        if(self.turn ==2):
          if((self.round+1)%4 == 1 or (self.round+1)%4 == 0):
            new_x = 5
            new_y = 1
            old_x = 5
            old_y = 0
          else:
            new_x = 7
            new_y = 3
            old_x = 8
            old_y = 3  

        else:
          reduce_dim = []
          for i in range(9):
            for j in range(9):
              reduce_dim.append(self.game_board[i][j])
              
          #if(self.turn >= 50): reduce_dim[80] = 1    
          env_out = env.negascout(reduce_dim)
          
          new_x = env_out[0][0]
          new_y = env_out[0][1]
          old_x = env_out[1][0]
          old_y = env_out[1][1]
          
        move_list = self.gen_step(self.chessID,self.game_board, 2)
        ######train#######
        board_list = []
        label_list = []
        flag = False
        for i in range(len(move_list)):
            (tox,toy) = move_list[i]["toPos"]
            (fromx,fromy) = move_list[i]["fromPos"]
            if((tox,toy) == (new_x,new_y) and (fromx,fromy) == (old_x,old_y)):
              label_list.append([1])
              flag = True
            else: label_list.append([0]) 
            
            self.game_board[fromx][fromy] = 0
            self.game_board[tox][toy] = 1
            
            temp_board = np.asarray(self.game_board,dtype=np.float32) *0.5
            #print temp_board
            '''
            for m in range(9):
              for n in range(9):
                if(temp_board[m][n] != 0): temp_board[m][n] -= 1.5
            '''  
            temp_board = np.reshape(temp_board,81)
            #print temp_board  
            #value = self.predict.eval(feed_dict={
            #    self.input_board: [temp_board], self.keep_prob: 1.0})
            
            board_list.append(temp_board)
            self.game_board[fromx][fromy] = 1
            self.game_board[tox][toy] = 0
          
        if(flag == False):
            print "ERROR!!"
            quit()

        label_list = np.asarray(label_list)
        self.train_step1.run(feed_dict={self.input_board: board_list, self.label: label_list, self.keep_prob: 0.5, self.discount: 1.0, self.scale:1.0})        
        train_accuracy = self.accuracy.eval(feed_dict={self.input_board: board_list, self.label: label_list, self.keep_prob: 1.0, self.discount: 1.0, self.scale:1.0})
        self.guess_count += train_accuracy
        
        #ep = self.round * 0.5 / self.terminate + 0.5
        ##random
        
        if(random.random() < 0.5 and self.turn >10):
            ran = random.randrange(6)
            #index = random.randrange(len(move_list))
            value_list = np.asarray(self.output.eval(feed_dict={self.input_board: board_list, self.keep_prob: 1.0}))
            value_list = value_list[:,0]           
            index = value_list.argsort()[-6:][::-1]
            ran = min(ran,len(index)-1)
            (new_x,new_y) = move_list[index[ran]]["toPos"]
            (old_x,old_y) = move_list[index[ran]]["fromPos"]
          ###################
        
        '''
        print "red:"
        print (old_x,old_y)
        print (new_x,new_y)
        '''
          
        self.game_board[old_x][old_y] = 0
        self.game_board[new_x][new_y] = 1
        self.rotated_board[8-old_x][8-old_y] = 0
        self.rotated_board[8-new_x][8-new_y] = -1
        
        index = self.chessID.index((old_x,old_y))
        self.chessID[index] = (new_x,new_y)
        
      if(self.old_turn == self.turn):
        #if(self.turn == 0): quit()
        #if(self.isSeparate(self.chessID) or self.turn >= 170):
        if(self.turn > 20):
          #print "############"
          #print "guess accuracy: ", self.guess_count / self.turn
          if(self.round == 50):
            print "end"
            saver = tf.train.Saver()
            saver.save(self.sess, './cc_model')
            saver.export_meta_graph('./cc_model.meta')
      
            
            quit()
          with open('output.csv', 'ab') as csvfile:
            spamwriter = csv.writer(csvfile)        
            spamwriter.writerow([self.round, self.guess_count / self.turn ])
          self.round += 1
          #self.PHASE3_FLAG = False
          self.game_board =[[0, 0, 0, 0, 0, -1, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 1, 0, 0, 0, 0, 0]]
          self.rotated_board=[[0, 0, 0, 0, 0, -1, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 1, 0, 0, 0, 0, 0]]
          self.chessID = [(0, 8), (0, 7), (1, 8), (0, 6), (1, 7), (2, 8), (0, 5), (1, 6), (2, 7), (3, 8), 
                    (8, 0), (7, 0), (8, 1), (6, 0), (7, 1), (8, 2), (5, 0), (6, 1), (7, 2), (8, 3)]
           
          self.turn = 1
          self.old_turn = 0
          self.guess_count = 0
          #print "end of one round"
          continue
        #print self.turn
        self.turn += 1
    '''    
    for i in range(2000):
      batch = data.train.next_batch(100)
  
      if i%100 == 0:
        train_accuracy = self.accuracy.eval(feed_dict={
            self.input_board:batch[0], self.label: batch[1], self.keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      self.train_step.run(feed_dict={self.input_board: batch[0], self.label: batch[1], self.keep_prob: 0.5})
    '''

    
  #opponent fix, self explore  
  def train_stage2(self):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('')
    if ckpt and ckpt.model_checkpoint_path:
        print "continue stage 2"
        saver.restore(self.sess,"./cc_model")
    else: 
        print "must train stage 1 first!"
        quit()
    
    while True:
      if(self.old_turn != self.turn):
        self.old_turn = self.turn
      ###blue
      if((self.turn +(self.round % 2))% 2 == 1 and self.blue_finished == False):
        if(self.red_finished == True ): self.diff += 1
        
        if(self.found_blue_phase3 == True):
          self.blue_end_layer -= 1
          if(self.blue_end_layer == 0): self.blue_finished = True
        
        else:
          if(self.PHASE3_FLAG == True):
            
            for x in range(20):
              self.rotated_chessID[x] = (8-self.chessID[19-x][0],8-self.chessID[19-x][1])
              
            for x in range(len(self.db3.keys())):
              if(self.is_same(self.rotated_chessID[10:20],self.db3[str(x)]['state'])):
                self.blue_end_layer = self.db3[str(x)]['layer']
                #print "blue", self.blue_end_layer
                self.blue_end_layer -= 1
                if(self.blue_end_layer == 0): self.blue_finished = True
                self.found_blue_phase3 = True
                break
          
          
          if(self.found_blue_phase3 == False):
          
                if(self.turn ==(self.round % 2 +1)):
                  if(self.round%4 == 1 or self.round%4 == 2):
                    new_x = 5
                    new_y = 1
                    old_x = 5
                    old_y = 0
                  else:
                    new_x = 7
                    new_y = 3
                    old_x = 8
                    old_y = 3
                  #print "blue"
                  #print (8-new_x,8-new_y)

                else:
                    
                    reduce_dim = []
                    for i in range(9):
                        for j in range(9):
                            reduce_dim.append(self.rotated_board[i][j])
                    if(self.PHASE3_FLAG == True): reduce_dim[0] = 1
                    #elif(self.turn >= 70): reduce_dim[80] = 1
                    #print reduce_dim
                    env_out = env.negascout(reduce_dim)
          
                    new_x = env_out[0][0]
                    new_y = env_out[0][1]
                    old_x = env_out[1][0]
                    old_y = env_out[1][1]
                    
                if(self.turn <= 20):
          
                      move_list = self.gen_step(self.chessID,self.game_board,1)
                      #######train#######
                      board_list = []
                      index_list = []
                      flag = False
                      #index1 = 0
                      for i in range(len(move_list)):
                        (tox,toy) = move_list[i]["toPos"]
                        (fromx,fromy) = move_list[i]["fromPos"]
                        if((tox,toy) == (8-new_x,8-new_y) and (fromx,fromy) == (8-old_x,8-old_y)):
                            index_list.append(i) 
                            flag = True
            
                        self.rotated_board[8-fromx][8-fromy] = 0
                        self.rotated_board[8-tox][8-toy] = 1
                        temp_board = np.asarray(self.rotated_board,dtype=np.float32)*0.5
   
                        temp_board = np.reshape(temp_board,81)

                        board_list.append(temp_board)
                        self.rotated_board[8-fromx][8-fromy] = 1
                        self.rotated_board[8-tox][8-toy] = 0
          
                      if(flag == False):
                        print "ERROR!!"
                        quit()

                      index_list = np.asarray(index_list)
                      self.blue_record["l"].append(index_list)
                      #self.blue_record["l"].append([1])
                      self.blue_record["b"].append(board_list)
                      
                '''
                print "blue:"
                print (8-old_x,8-old_y)
                print (8-new_x,8-new_y)  
                '''
                self.game_board[8-old_x][8-old_y] = 0
                self.game_board[8-new_x][8-new_y] = -1
        
                self.rotated_board[old_x][old_y] = 0
                self.rotated_board[new_x][new_y] = 1
        
                index = self.chessID.index((8-old_x,8-old_y))
                self.chessID[index] = (8-new_x,8-new_y)
                    ##############
      
      
      
      #####red  
      elif((self.turn +(self.round % 2))% 2 == 0 and self.red_finished == False):
        
        if(self.blue_finished == True): self.diff -= 1
        
        if(self.found_red_phase3 == True):
          self.red_end_layer -=1
          if(self.red_end_layer == 0): self.red_finished = True
      
        else:
          if(self.PHASE3_FLAG == True):

            for x in range(len(self.db3.keys())):
                if(self.is_same(self.chessID[10:20],self.db3[str(x)]['state'])):
                    self.red_end_layer = self.db3[str(x)]['layer']
                    #print "red", self.red_end_layer
                    self.red_end_layer -= 1
                    if(self.red_end_layer == 0): self.red_finished = True
                    self.found_red_phase3 = True
                    break
          if(self.found_red_phase3 == False):    
            ##################
            if(self.turn <= 20):
                
                board_list = []
                index_list = []  
                   
                move_list = self.gen_step(self.chessID,self.game_board,2)
                #print move_list
                for i in range(len(move_list)):
                    
                    (tox,toy) = move_list[i]["toPos"]
                    (fromx,fromy) = move_list[i]["fromPos"]
            
                    self.game_board[fromx][fromy] = 0
                    self.game_board[tox][toy] = 1
            
                    temp_board = np.asarray(self.game_board,dtype=np.float32) *0.5
                    temp_board = np.reshape(temp_board,81)

                    board_list.append(temp_board)
                    self.game_board[fromx][fromy] = 1
                    self.game_board[tox][toy] = 0
                   
                list = np.asarray(self.output.eval(feed_dict={self.input_board: board_list, self.keep_prob: 1.0}))
                #print list
                list = list[:,0]   
                '''
                threshold = 0.2 - self.round *0.2 / self.terminate 
                if(random.random() < threshold and self.turn >6):
                    index = random.randrange(len(move_list))
                    (new_x,new_y) = move_list[index]["toPos"]
                    (old_x,old_y) = move_list[index]["fromPos"]
                    #index += 100
                else:
                '''

                
                index = list.argsort()[::-1][0]
                (new_x,new_y) = move_list[index]["toPos"]
                (old_x,old_y) = move_list[index]["fromPos"]

                ################################3
                index_list.append(index)
                #print index
                index_list = np.asarray(index_list)
                self.red_record["l"].append(index_list)

                self.red_record["b"].append(board_list)
                
            #################
            else:
                reduce_dim = []
                for i in range(9):
                    for j in range(9):
                        reduce_dim.append(self.game_board[i][j])
                if(self.PHASE3_FLAG == True): reduce_dim[0] = 1
                #elif(self.turn >= 70): reduce_dim[80] = 1
                #print reduce_dim
                #reduce_dim[1] = layer
                env_out = env.negascout(reduce_dim)
          
                new_x = env_out[0][0]
                new_y = env_out[0][1]
                old_x = env_out[1][0]
                old_y = env_out[1][1]
            
            
            '''
            print "red:"
            print (old_x,old_y)
            print (new_x,new_y)
            '''
            
            
            self.game_board[old_x][old_y] = 0
            self.game_board[new_x][new_y] = 1
            self.rotated_board[8-old_x][8-old_y] = 0
            self.rotated_board[8-new_x][8-new_y] = -1
        
            index = self.chessID.index((old_x,old_y))
            self.chessID[index] = (new_x,new_y)
      
      
      if(self.red_finished == True and self.blue_finished == True):
          #print self.diff
          #if(self.diff >= 5): quit()
          with open('output1.csv', 'ab') as csvfile:
            spamwriter = csv.writer(csvfile)        
            spamwriter.writerow([self.round, self.diff ])
          if(self.round == 400):
            print "end"
            saver = tf.train.Saver()
            saver.save(self.sess, './cc_model')
            saver.export_meta_graph('./cc_model.meta')
            quit()
          
          
          #diff:red win -> + ; blue win-> -
          
          blue_len = len(self.blue_record["l"])
          red_len = len(self.red_record["l"])
          if (self.diff == 1): self.diff = -1
          if(self.diff < 0):
            
            for i in range(red_len):

                self.train_step2.run(feed_dict={self.input_board: self.red_record["b"][i], self.label: [[0]], self.col_index: self.red_record["l"][i], 
                                       self.keep_prob: 0.5, self.discount: pow(self.gramma,(red_len - i -1)), self.scale:min(10,abs(self.diff))/10.0})

            for i in range(blue_len):
                self.train_step2.run(feed_dict={self.input_board: self.blue_record["b"][i], self.label:[[1]], self.col_index: self.blue_record["l"][i], 
                                       self.keep_prob: 0.5, self.discount: pow(self.gramma,(blue_len - i -1)), self.scale:min(10,abs(self.diff))/10.0})

          if(self.diff > 0):
            for i in range(red_len):

                self.train_step2.run(feed_dict={self.input_board: self.red_record["b"][i], self.label: [[1]], self.col_index: self.red_record["l"][i], 
                                       self.keep_prob: 0.5, self.discount: pow(self.gramma,(red_len - i -1)), self.scale:min(10,abs(self.diff))/10.0})

            for i in range(blue_len):
                
                self.train_step2.run(feed_dict={self.input_board: self.blue_record["b"][i], self.label:[[0]], self.col_index: self.blue_record["l"][i], 
                                       self.keep_prob: 0.5, self.discount: pow(self.gramma,(blue_len - i -1)), self.scale:min(10,abs(self.diff))/10.0})
          
          
          self.round += 1
          #self.PHASE3_FLAG = False
          self.game_board =[[0, 0, 0, 0, 0, -1, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 1, 0, 0, 0, 0, 0]]
          self.rotated_board=[[0, 0, 0, 0, 0, -1, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 1, 0, 0, 0, 0, 0]]
          self.chessID = [(0, 8), (0, 7), (1, 8), (0, 6), (1, 7), (2, 8), (0, 5), (1, 6), (2, 7), (3, 8), 
                    (8, 0), (7, 0), (8, 1), (6, 0), (7, 1), (8, 2), (5, 0), (6, 1), (7, 2), (8, 3)]
           
          self.turn = 1
          self.old_turn = 0
          self.PHASE3_FLAG = False
          self.found_blue_phase3 = False
          self.found_red_phase3 = False
          self.blue_finished = False
          self.red_finished = False
          self.diff = 0
          self.blue_record = {"l":[],"b":[]}
          self.red_record = {"l":[],"b":[]}
          #self.red_bad = {"l":[],"b":[]}
          #print "end of one round"
          continue

      
      if(self.old_turn == self.turn):
        #print self.turn
        if(self.turn >= 150): 
          self.blue_finished = True
          self.red_finished = True
          self.diff = 0

          
        if(self.isSeparate(self.chessID)):
          self.PHASE3_FLAG = True
          #print "stop"        
        self.turn += 1
  '''  
  def negaScoutAB(self,depth,a,b,chessid,board):
    side = (self.tempDepth - depth) % 2 # 0:AI 1:opponent
    if(self.PHASE3_FLAG == True and side == 0):
        flag = True
        for i in range(10,20):
          if(chessid[i] not in self.win):
            flag = False
            break
        if(flag == True): return 19980 + depth     

    x = self.key32 & int("FFFFF",16)
    score = 66666
    try: 
      element = self.TRANSTABLE[(side,x)]

      if(element["depth"] >= depth and element["checksum"] == self.key64):
        if(element["type"] == 1):
          score = element["eval"]
        elif(element["type"] == 2 and element["eval"] >= b):
          score = element["eval"]
        elif(element["type"] == 3 and element["eval"] <= a):
          score = element["eval"]
    except KeyError: pass


    if(score != 66666): return score
  
    if(depth <= 0):

      #print board
      tempboard = np.asarray(board,dtype=np.float32) * 0.5
      tempboard = np.reshape(tempboard,81)
      #print tempboard
      score = self.output.eval(feed_dict={self.input_board: [tempboard], self.keep_prob: 1.0})[0]

      x = self.key32 & int("FFFFF",16)
      element = {"checksum":self.key64,"type":1,"eval":score,"depth":depth}
      self.TRANSTABLE[(side,x)] = element

      return score

    temp_movelist = [] #ten elements for the current
    move_list = [] #filter out some moves

    stack = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    if(side == 0):
      for i in range(10,20):
      
        #result = []
        #stack = []
        (x,y) = chessid[i]
        stack[0]=(x,y)
        result=[(x,y)]
        stack_len = 1;
        while (stack_len > 0):        
          (x1,y1) = stack[stack_len-1]
          stack_len -= 1

          if(x1 < 2): 
            flag_x1 = False
            flag_x2 = True
          elif(x1 > 6):
            flag_x1 = True
            flag_x2 = False
          else:
            flag_x1 = True
            flag_x2 = True
          if(y1 < 2): 
            flag_y1 = False
            flag_y2 = True
          elif(y1 > 6):
            flag_y1 = True
            flag_y2 = False
          else:
            flag_y1 = True
            flag_y2 = True
        
          if(flag_x1 == True and flag_y1 == True  and board[x1-1][y1-1] != 0 and board[x1-2][y1-2] == 0 and ((x1-2,y1-2) not in result)):
            #stack.append((x1-2,y1-2))
            stack[stack_len] = (x1-2,y1-2)
            stack_len += 1
            result.append((x1-2,y1-2))
          if(flag_x2 == True and flag_y2 == True and board[x1+1][y1+1] != 0 and board[x1+2][y1+2] == 0 and   ((x1+2,y1+2) not in result)):
            #stack.append((x1+2,y1+2))
            stack[stack_len] = (x1+2,y1+2)   
            stack_len += 1
          
            result.append((x1+2,y1+2))

          if(flag_x2 == True and board[x1+1][y1] != 0 and board[x1+2][y1] == 0 and  ((x1+2,y1) not in result)):
            #stack.append((x1+2,y1))
            stack[stack_len] = (x1+2,y1)
            stack_len += 1
            result.append((x1+2,y1))
          if(flag_y1 == True  and board[x1][y1-1] != 0 and board[x1][y1-2] == 0 and ((x1,y1-2) not in result)):
            #stack.append((x1,y1-2))  
            stack[stack_len] = (x1,y1-2)    
            stack_len += 1            
            result.append((x1,y1-2))
          if(flag_y2 == True and board[x1][y1+1] != 0 and board[x1][y1+2] == 0 and ((x1,y1+2) not in result)):
            #stack.append((x1,y1+2))  
            stack[stack_len] = (x1,y1+2)            
            stack_len += 1          
            result.append((x1,y1+2))
          if(flag_x1 == True and board[x1-1][y1] != 0 and board[x1-2][y1] == 0 and ((x1-2,y1) not in result)):
            #stack.append((x1-2,y1))  
            stack[stack_len] = (x1-2,y1)  
            stack_len += 1          
            result.append((x1-2,y1))
  
        result.remove((x,y))

        if(depth != 1 or len(result) == 0):

          if(x>=1 and x <=9 and y >=1 and y <=9 and board[x-1][y-1] == 0):
            result.append((x-1,y-1))  
          if(x >=-1 and x <=7 and y >=-1 and y <=7 and board[x+1][y+1] == 0):
            result.append((x+1,y+1))
          if(x >=0 and x <=8 and y >=-1 and y <=7 and board[x][y+1] == 0):
            result.append((x,y+1))
          if(x >=1 and x <=9 and y >=0 and y <=8 and board[x-1][y] == 0):
            result.append((x-1,y))
      
        temp_movelist.append(result)
    else:
      for i in range(10):
        (x,y) = chessid[i]
        #if(x >= 0 and x <= 8 and y >= 0 and y <= 8):

        #result = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #result = []
        #stack = []    
        #count = 0
        stack[0]=(x,y)
        #result[0] = (x,y)
        result=[(x,y)]
        stack_len = 1
        #count += 1

        while (stack_len > 0):
          (x1,y1) = stack[stack_len-1]
          stack_len -= 1          
          if(x1 < 2): 
            flag_x1 = False
            flag_x2 = True
          elif(x1 > 6):
            flag_x1 = True
            flag_x2 = False
          else:
            flag_x1 = True
            flag_x2 = True
          if(y1 < 2): 
            flag_y1 = False
            flag_y2 = True
          elif(y1 > 6):
            flag_y1 = True
            flag_y2 = False
          else:
            flag_y1 = True
            flag_y2 = True
          if(flag_x1 == True and flag_y1 == True and board[x1-1][y1-1] != 0 and board[x1-2][y1-2] == 0 and ((x1-2,y1-2) not in result)):
            stack[stack_len]=(x1-2,y1-2)
            stack_len += 1
            result.append((x1-2,y1-2))
            #result[count] = (x1-2,y1-2)
            #count += 1
          if(flag_x2 == True and flag_y2 == True and board[x1+1][y1+1] != 0 and board[x1+2][y1+2] == 0 and   ((x1+2,y1+2) not in result)):
            #stack.append((x1+2,y1+2))
            stack[stack_len]=(x1+2,y1+2)
            stack_len += 1
            result.append((x1+2,y1+2))
            
            #result[count] = (x1+2,y1+2)
            #count += 1
          if(flag_x2 == True and board[x1+1][y1] != 0 and board[x1+2][y1] == 0 and  ((x1+2,y1) not in result)):
            #stack.append((x1+2,y1))
            stack[stack_len]=(x1+2,y1)
            stack_len += 1            
            result.append((x1+2,y1))
            #result[count] = (x1+2,y1)
            #count += 1
          if(flag_y1 == True  and board[x1][y1-1] != 0 and board[x1][y1-2] == 0 and ((x1,y1-2) not in result)):
            #stack.append((x1,y1-2))  
            stack[stack_len]=(x1,y1-2)
            stack_len += 1            
            result.append((x1,y1-2))
            #result[count]=(x1,y1-2)
            #count += 1

          if(flag_y2 == True and board[x1][y1+1] != 0 and board[x1][y1+2] == 0 and ((x1,y1+2) not in result)):
              #stack.append((x1,y1+2))  
              stack[stack_len]=(x1,y1+2)
              stack_len += 1
              result.append((x1,y1+2))
              #result[count]=(x1,y1+2)
              #count += 1
          if(flag_x1 == True and board[x1-1][y1] != 0 and board[x1-2][y1] == 0 and ((x1-2,y1) not in result)):
              #stack.append((x1-2,y1))  
              stack[stack_len]=(x1-2,y1)
              stack_len += 1
              result.append((x1-2,y1))
              #result[count]=(x1-2,y1)
              #count += 1
  
        result.remove((x,y))
        #if(side == 0): print result  
  
        if(depth != 1 or len(result) == 0):
          ####move
          if(x >=1 and x <=9 and y>=1 and y<=9 and board[x-1][y-1] == 0):
            result.append((x-1,y-1)) 
            #result[count] = (x-1,y-1)
            #count += 1
          if(x>=-1 and x <=7 and y >=-1 and y <=7 and board[x+1][y+1] == 0):
            result.append((x+1,y+1))
            #result[count] = (x+1,y+1)
            #count += 1
          if(x >=-1 and x<=7 and y >=0 and y <=8 and board[x+1][y] == 0):
            result.append((x+1,y))
            #result[count] = (x+1,y)
            #count += 1
          if(x >=0 and x <=8 and y >=1 and y <=9 and board[x][y-1] == 0):
            result.append((x,y-1))
            #result[count] = (x,y-1)
            #count += 1
        
        temp_movelist.append(result) 
    

    maxscore = -99
    if(depth == 1): move_list.append({})
    elif(self.PHASE3_FLAG == True and side == 1 ): move_list.append({})
    for i in range(10):
    
      for j in range(len(temp_movelist[i])):
        (x,y) = temp_movelist[i][j]

        if(side == 0):
          old = chessid[i+10]
          if(depth == 1):
            score = self.TrainingBoard[8-x][8-y] - self.TrainingBoard[8-old[0]][8-old[1]]
            if(score > maxscore):
              maxscore = score
              move_list[0] = {"toPos": (x,y), "fromPos": old, "score":0}
          else:
            if((y-x) >= (old[1]-old[0]) and self.TrainingBoard[8-x][8-y] != -99):
              move_list.append({"toPos": (x,y), "fromPos": old, "score":0})

        else:
          old = chessid[i]
          if(depth == 1 or self.PHASE3_FLAG == True ):
            score = self.TrainingBoard[x][y] - self.TrainingBoard[old[0]][old[1]]
            if(score > maxscore):
              maxscore = score
              move_list[0] = {"toPos": (x,y), "fromPos": old, "score":0}
          else:
            if((y-x) <= (old[1]-old[0]) and self.TrainingBoard[x][y] != -99):
              move_list.append({"toPos": (x,y), "fromPos": old, "score":0})
    

    for i in range(len(move_list)):
      nfrom = move_list[i]['fromPos'][0] *9 + move_list[i]['fromPos'][1]
      nto = move_list[i]['toPos'][0]*9 + move_list[i]['toPos'][1]
      move_list[i]['score'] = self.history_table[nfrom][nto]
   
    move_list.sort(key=lambda x:x['score'], reverse=True)

    if(depth == self.tempDepth and self.tempDepth > 3):

      for i in range(len(move_list)):
      
        if(self.BESTMOVE[-1][0] == move_list[i]['toPos'] and
           self.BESTMOVE[-1][1] == move_list[i]['fromPos']):
          temp = move_list[0]
          move_list[0] = move_list[i]
          move_list[i] = temp
          break
  
    bestmove = -1
    alpha = a
    beta = b
    is_exact = False  


    for i in range(len(move_list)):

      (tox,toy) = move_list[i]['toPos']
      (fromx,fromy) = move_list[i]['fromPos']

      id = chessid.index((fromx,fromy))
      self.key32 = self.key32 ^ self.hash_key32[id][fromx][fromy]
      self.key64 = self.key64 ^ self.hash_key64[id][fromx][fromy]
  
      self.key32 = self.key32 ^ self.hash_key32[id][tox][toy]
      self.key64 = self.key64 ^ self.hash_key64[id][tox][toy]

      chessid[id] = (tox,toy)
      board[fromx][fromy] = 0
      if(side == 0):board[tox][toy] = 1
      else: board[tox][toy] = -1

      t = -self.negaScoutAB(depth-1,-beta,-alpha,chessid,board)

      if(t > alpha and t < b and i >0):
        alpha = -self.negaScoutAB(depth-1,-b,-t,chessid,board)
        is_exact = True
        if(depth == self.tempDepth):
 
          result_move = [(tox,toy),(fromx,fromy)]
          self.BESTMOVE.append(result_move)
        bestmove = i

      self.key32 = self.key32 ^ self.hash_key32[id][tox][toy]
      self.key64 = self.key64 ^ self.hash_key64[id][tox][toy]
  
      self.key32 = self.key32 ^ self.hash_key32[id][fromx][fromy]
      self.key64 = self.key64 ^ self.hash_key64[id][fromx][fromy]

      chessid[id] = (fromx,fromy)
      board[tox][toy] = 0
      if(side == 0): board[fromx][fromy] =1
      else: board[fromx][fromy] =-1   

      if(alpha < t):
        is_exact = True

        alpha = t
      
        if(depth == self.tempDepth):

          result_move = [(tox,toy),(fromx,fromy)]
          self.BESTMOVE.append(result_move)

      if(alpha >= b):

        x = self.key32 & int("FFFFF",16)
        self.TRANSTABLE[(side,x)] = {"checksum":self.key64,"type":2,"eval":alpha,"depth":depth}

        self.history_table[fromx *9 + fromy][tox*9 + toy] += 2 ** (depth + 1)

        return alpha
      beta = alpha +1
  
    if(bestmove != -1): 
      (tox,toy) = move_list[bestmove]['toPos']
      (fromx,fromy) = move_list[bestmove]['fromPos']

      self.history_table[fromx *9 + fromy][tox*9 + toy] += 2 ** (depth + 1)

    if(is_exact == True):

      x = self.key32 & int("FFFFF",16)
      self.TRANSTABLE[(side,x)] = {"checksum":self.key64,"type":1,"eval":alpha,"depth":depth}
   
    else:

      x = self.key32 & int("FFFFF",16)
      self.TRANSTABLE[(side,x)] = {"checksum":self.key64,"type":3,"eval":alpha,"depth":depth}

    return alpha  
  '''  
    
  def play(self,layer,type):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('')
    if ckpt and ckpt.model_checkpoint_path:
        print "testing"
        saver.restore(self.sess,"./cc_model")
    else: 
        print "must train stage 1 first!"
        quit()
    
    while True:
      if(self.old_turn != self.turn):
        self.old_turn = self.turn
      ###blue
      if((self.turn +(self.round % 2))% 2 == 1 and self.blue_finished == False):
        if(self.red_finished == True ): self.diff += 1
        
        if(self.found_blue_phase3 == True):
          self.blue_end_layer -= 1
          if(self.blue_end_layer == 0): self.blue_finished = True
        
        else:
          if(self.PHASE3_FLAG == True):
            
            for x in range(20):
              self.rotated_chessID[x] = (8-self.chessID[19-x][0],8-self.chessID[19-x][1])
              
            for x in range(len(self.db3.keys())):
              if(self.is_same(self.rotated_chessID[10:20],self.db3[str(x)]['state'])):
                self.blue_end_layer = self.db3[str(x)]['layer']
                #print "blue", self.blue_end_layer
                self.blue_end_layer -= 1
                if(self.blue_end_layer == 0): self.blue_finished = True
                self.found_blue_phase3 = True
                break
            print "aaaaa"
          
          if(self.found_blue_phase3 == False):
          
                if(self.turn ==(self.round % 2 +1)):
                  if(random.random() < 0.5):
                    new_x = 5
                    new_y = 1
                    old_x = 5
                    old_y = 0
                  else:
                    new_x = 7
                    new_y = 3
                    old_x = 8
                    old_y = 3

                else:
                    
                    reduce_dim = []
                    for i in range(9):
                        for j in range(9):
                            reduce_dim.append(self.rotated_board[i][j])
                    if(self.PHASE3_FLAG == True): reduce_dim[0] = 1
                    reduce_dim[1] = layer
                    #reduce_dim[80] = type
                    #elif(self.turn >= 70): reduce_dim[80] = 1
                    #print reduce_dim
                    if(type == 1): 
                       env_out = env_def.negascout(reduce_dim)
                    elif(type == 2):
                       env_out = env_off.negascout(reduce_dim)
                    else:
                       env_out = env.negascout(reduce_dim)
                       
                    new_x = env_out[0][0]
                    new_y = env_out[0][1]
                    old_x = env_out[1][0]
                    old_y = env_out[1][1]
                    
                
                     
                      
                
                print "blue:"
                print (8-old_x,8-old_y)
                print (8-new_x,8-new_y)
                
                self.game_board[8-old_x][8-old_y] = 0
                self.game_board[8-new_x][8-new_y] = -1
        
                self.rotated_board[old_x][old_y] = 0
                self.rotated_board[new_x][new_y] = 1
        
                index = self.chessID.index((8-old_x,8-old_y))
                self.chessID[index] = (8-new_x,8-new_y)
                    ##############
      
      
      
      #####red  
      elif((self.turn +(self.round % 2))% 2 == 0 and self.red_finished == False):
        
        if(self.blue_finished == True): self.diff -= 1
        
        if(self.found_red_phase3 == True):
          self.red_end_layer -=1
          if(self.red_end_layer == 0): self.red_finished = True
      
        else:
          if(self.PHASE3_FLAG == True):

            for x in range(len(self.db3.keys())):
                if(self.is_same(self.chessID[10:20],self.db3[str(x)]['state'])):
                    self.red_end_layer = self.db3[str(x)]['layer']
                    #print "red", self.red_end_layer
                    self.red_end_layer -= 1
                    if(self.red_end_layer == 0): self.red_finished = True
                    self.found_red_phase3 = True
                    break
            print "bbbbb"
          if(self.found_red_phase3 == False):    
            ##################
            
            
            if(self.turn <= 20):
                board_list = []
                index_list = []  
                   
                move_list = self.gen_step(self.chessID,self.game_board,2)
                #print move_list
                for i in range(len(move_list)):
                    
                    (tox,toy) = move_list[i]["toPos"]
                    (fromx,fromy) = move_list[i]["fromPos"]
            
                    self.game_board[fromx][fromy] = 0
                    self.game_board[tox][toy] = 1
            
                    temp_board = np.asarray(self.game_board,dtype=np.float32) *0.5
                    temp_board = np.reshape(temp_board,81)

                    board_list.append(temp_board)
                    self.game_board[fromx][fromy] = 1
                    self.game_board[tox][toy] = 0
                   
                list = np.asarray(self.output.eval(feed_dict={self.input_board: board_list, self.keep_prob: 1.0}))
                #print list
                list = list[:,0]      

                index = list.argsort()[::-1][0]

                #index_list.append(index)
                #print index
                #index_list = np.asarray(index_list)
                #self.red_record["l"].append(index_list)

                #self.red_record["b"].append(board_list)
                (new_x,new_y) = move_list[index]["toPos"]
                (old_x,old_y) = move_list[index]["fromPos"]

            else:
                reduce_dim = []
                for i in range(9):
                    for j in range(9):
                        reduce_dim.append(self.game_board[i][j])
                if(self.PHASE3_FLAG == True): reduce_dim[0] = 1
                #elif(self.turn >= 70): reduce_dim[80] = 1
                #print reduce_dim
                reduce_dim[1] = layer
                env_out = env_def.negascout(reduce_dim)
          
                new_x = env_out[0][0]
                new_y = env_out[0][1]
                old_x = env_out[1][0]
                old_y = env_out[1][1]
            '''    
            #################
            if(self.turn ==2 - (self.round % 2)):
                if(random.random() < 0.5):
                    new_x = 7
                    new_y = 3
                    old_x = 8
                    old_y = 3  
                else:
                    
                    new_x = 5
                    new_y = 1
                    old_x = 5
                    old_y = 0
            '''
            
            print "red:"
            print (old_x,old_y)
            print (new_x,new_y)
            
            self.game_board[old_x][old_y] = 0
            self.game_board[new_x][new_y] = 1
            self.rotated_board[8-old_x][8-old_y] = 0
            self.rotated_board[8-new_x][8-new_y] = -1
        
            index = self.chessID.index((old_x,old_y))
            self.chessID[index] = (new_x,new_y)
      
      
      if(self.red_finished == True and self.blue_finished == True):
          print "########"
          #if(self.diff >= 5): quit()
          with open('test.csv', 'ab') as csvfile:
            spamwriter = csv.writer(csvfile)        
            spamwriter.writerow([self.round, self.diff ])
          if(self.round == 40):
            print "end"
            '''
            saver = tf.train.Saver()
            saver.save(self.sess, './cc_model')
            saver.export_meta_graph('./cc_model.meta')
            '''
            quit()
          
          
          #diff:red win -> + ; blue win-> -
          
          
          self.round += 1
          #self.PHASE3_FLAG = False
          self.game_board =[[0, 0, 0, 0, 0, -1, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 1, 0, 0, 0, 0, 0]]
          self.rotated_board=[[0, 0, 0, 0, 0, -1, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, -1, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, -1, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, -1], 
                 [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 0, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 0, 0, 0, 0, 0, 0], 
                 [1, 1, 1, 1, 0, 0, 0, 0, 0]]
          self.chessID = [(0, 8), (0, 7), (1, 8), (0, 6), (1, 7), (2, 8), (0, 5), (1, 6), (2, 7), (3, 8), 
                    (8, 0), (7, 0), (8, 1), (6, 0), (7, 1), (8, 2), (5, 0), (6, 1), (7, 2), (8, 3)]
           
          self.turn = 1
          self.old_turn = 0
          self.PHASE3_FLAG = False
          self.found_blue_phase3 = False
          self.found_red_phase3 = False
          self.blue_finished = False
          self.red_finished = False
          self.diff = 0
          #self.red_bad = {"l":[],"b":[]}
          #print "end of one round"
          continue

      
      if(self.old_turn == self.turn):
        #print self.turn
        if(self.turn >= 150): 
          self.blue_finished = True
          self.red_finished = True
          self.diff = 0

          
        if(self.isSeparate(self.chessID)):
          self.PHASE3_FLAG = True
          #print "stop"        
        self.turn += 1


