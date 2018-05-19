# -*- coding: utf-8 -*-
"""
A feed-forward MLP classifies facial action units (AUs) from the Facial Action Coding System
based on displacements of facial landmarks. The approach is based on the paper, 
"Recognizing Action Units for Facial Expression Analysis" by Tian, et al.

The dlib library extracts facial landmarks from the Facial image sequences from the CK+ database.
Distances between key points, such as between the eyelids and separation of the eyebrows,
are calculated. The beginning of each CK+ is neutral and the end is the expression at
maximum intensity. The change in distance is calculated and fed into the neural network. 

The algorithm is a multiclass binary classifier and returns AUs. Because 
basic emotions may be represented by certain facial actions, the determined AUs
point to the primary basic emotion. Two neural networks operate on the upper and lower
part of the face. On the test samples, the upper AUs have 78% accuracy and the lower AUs
have 84% accuracy. The accuracy is a little less than the paper by Tian, but this script uses
one database, whereas Tian used two. Possible reasons for misclassifying samples are the angle of the head which
shifts the coordinates, some AUs occur often in pairs, and limited data. 

dict_upper = ['AU1: Inner Brow Raiser','AU2: Outer Brow Raiser','AU4: Brow Lowerer','AU5: Upper Lid Raiser','AU6: Cheek Raiser','AU7: Lid Tightener']
dict_lower = ['AU9: Nose Wrinkler', 'AU10: Upper Lip Raiser', 'AU12: Lip Corner Puller', 'AU15: Lip Corner Depressor',  'AU17: Chin Raiser',  'AU20: Lip Stretcher',  'AU23: Lip Tightener','AU24: Lip Pressor', 'AU25: Lips Part',  'AU26: Jaw Drop',  'AU27: Mouth Stretch']

A video demo is available at joshlamstein.com. 
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import facs_helper
import facs_preparation

#%% INITIALIZE VARIABLES
###################################################################

facsList = [  1.,   2.,   4.,   5.,   6.,   7.,   9.,  10.,  11.,  12.,  13.,
        14.,  15.,  16.,  17.,  18.,  20.,  21.,  22.,  23.,  24.,  25.,
        26.,  27.,  28.,  29.,  30.,  31.,  34.,  38.,  39.,  43.,  44.,
        45.,  54.,  61.,  62.,  63.,  64.]
facsPaper = [5., 9., 12., 15., 16.,20.,23., 24.] # Kotsia FAU
facsTop = [1.,2.,4.,5.,6.,7.]
facsBtm = np.array([9., 10., 12., 15., 17., 20., 23.,24., 25., 26., 27.])
# get rid of facs 24, 26, 23, 10

predictor_path = '/Users/joshualamstein/Desktop/pyFiles/shape_predictor_68_face_landmarks.dat'
faces_folder_path = '/Users/joshualamstein/Desktop/CK+/cohn-kanade-images/**/*/'
#landmark_path = '/Users/joshualamstein/Desktop/CK+/Landmarks/S022/003/S022_003_00000005_landmarks.txt'
facs_folder = '/Users/joshualamstein/Desktop/CK+/FACS/**/*/'
emotion_folder = '/Users/joshualamstein/Desktop/CK+/Emotion/'

# Boolean to select whether you want to train for top FACS, bottom FACS, or facs from Kotsia's paper. 
paperBool = False
topBool = True
btmBool = False

#%% FACIAL FEATURES
"""
        facialMotion: Input data, upper face changes in distance from landmarks
        facialMotionLower: Input data, lower face changes in distance from landmarks
"""

# Get changes in distance of key facial features such as eyebrow distance or distance between
# parted lips. 
facial_features = facs_preparation.features(faces_folder_path, predictor_path)
facial_features.get_features()
facialMotion = facial_features.facialMotion 
facialMotionLower = facial_features.facialMotionLower 

#%% FACS
###################################################################
"""
        inten: intensity of facial expression, may be used to weight data
        intenT: intensity of top facial expressions, selected from Tian's paper
        intenB: intensity of bottom facial expressions, selected from Tian's paper
        facs: all facial action units in CK+
        facsT: top facial action units
        facsB: bottom facial action units
        facsPath: path of all facial action units

"""

# Gets AUs of facial sequences
folder = facs_helper.facsFolder(facs_folder, facsList, facsTop, facsBtm, facsPaper)
folder.process()

inten = folder.inten 
intenT = folder.intenT
intenB = folder.intenB
facs= folder.facs
facsT = folder.facsT
facsB= folder.facsB
facsPath = folder.facsPath

print('Finished facs')

#%% BASIC EMOTIONS
"""
This class sees how well the ground truth AUs predict the ground truth emotions. They have a 76% match.
For the classification, I referred to Facial Expression Recognition in Image Sequences Using Geometric Deformation Features
and Support Vector Machines by Kotsia and https://en.wikipedia.org/wiki/Facial_Action_Coding_System.

In the CK+ dataset, the labels for emotions are:
Basic Emotions: 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise

  Basic emotions and their associated AUs:
  
  # Happiness - AU 6, 12
  # Sadness - AU 1,4,15
  # Surprise - AU 1,2,5,26
  # Fear - AU 1,2,4,5,7,20,26
  # Anger - AU 4,5,7,23
  # Disgust - AU 9,15,16 
  # Contempt - AU 12,14
  
  emotions: emotions associated with sequence
  emotionsPath: path to emotions
  emoMatch: Returns the emotion number if matched, -1 if no match
  matchBool: True for match, false for no
  countM, E: counts the different emotions that matched
  match_miss_count: counts which predictions didn't match
  emo_miss_count: counts which predictions didn't match in ground truth
  
"""
###################################################################

# Get emotion of samples. Emotions may be found after AUs are predicted. 
emo = facs_helper.emoFolder(emotion_folder, facs_folder)
emo.process()
emotions = emo.emotions
emotionPath = emo.emotionPath

comp = facs_helper.compare(emotions,facsT, facsB)
emoMatch, matchBool = comp.match()

# Using the true values of AUs and basic emotions, 
# check the accuracy of how the AUs determine emotions. 
# Happiness is very clear based on AUs, but anger and 
# disgust have similar AUs and they can be confused. 
uniqueM, countM = np.unique(emoMatch, return_counts=True)
uniqueE, countE = np.unique(emotions,return_counts=True)
emoMatch[emoMatch==-2]=-1
# This checks where the the predicted matches against the ground truth emotions.
ii = np.where(emoMatch != emotions)

emoM,match_miss_count = np.unique(emoMatch[ii],return_counts=True)
emoList,emo_miss_count = np.unique(emotions[ii],return_counts=True)


#%% Set up data
###################################################################

# Write 'top' or 'bot' to select whether the model should train
# for AUs in the top of the face or the bottom of the face. 
# Mind that you have had topBool and botBool properly set when you processed data!
selectData = 'top'

if selectData == 'top':
    facialSelect = facialMotion
    # You can select which top AUs you train for with facsLimit. 
    facsLimit = 5
    facsLevel = facsT[:,:facsLimit]
    facsUse = facsTop[:facsLimit]
    weight_inten = intenT[:,:facsLimit]

elif selectData == 'bot':

    useBotIdx = np.array([0,2,3,4,5,8,10], dtype = np.int32) # You can select which AUs you train for with this array. 
    facialSelect = facialMotionLower
    facsLevel = facsB[:,useBotIdx]
    facsUse = facsBtm[useBotIdx]
    weight_inten= intenB[:,useBotIdx]

# Randomize samples
train_x, test_x, train_y, test_y, sw_train,sw_test= train_test_split(
    facialSelect, facsLevel, weight_inten, test_size=0.25, random_state=42) #0.25, 42

numFacs = len(facsUse)

# number of neurons in each layer
input_shape = facialSelect.shape
output_shape = [facialSelect.shape[0],numFacs]
num_samples = facialSelect.shape[0]
n_input = facialSelect.shape[1]
n_hidden_1 = 256 #256 neurons
n_hidden_2 = 256
n_hidden_3 = 256
n_hidden_4 = 256
n_hidden_5 = 256
n_hidden_6 = 256
n_output = numFacs

#%% Feed forward MLP NN
###################################################################

LOGDIR = "/tmp/FACS/zappa/upper/51/"
savepath = "/Users/joshualamstein/Desktop/pyFiles/save_tf_upper_zappa_11/"


# Function for fully connected layer
def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        print("w",w.dtype)
        print("b",b.dtype)
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act
   
# Neural Network Model
def facs_model(learning_rate, scale_class_weight,use_two_fc, use_three_fc, use_four_fc, use_five_fc, use_six_fc, use_seven_fc, hparam):
    config = tf.ConfigProto(graph_options=tf.GraphOptions(
    optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    tf.reset_default_graph()
    sess = tf.Session("", config = config)

  # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, [None, n_input], name = "x")
    y = tf.placeholder(tf.float32, [None, n_output], name = "labels")
    sw = tf.placeholder(tf.float32, [None, n_output], name = 'intensity_weights')

    # Main function compares the number of FCs for performance. 
    if use_two_fc:
        fc1 = fc_layer(x, n_input, n_hidden_1, "fc1")
        relu = tf.nn.relu(fc1)
        tf.summary.histogram("fc1/relu", relu)
        logits = fc_layer(relu, n_hidden_1, n_output, "fc2")
    elif use_three_fc: 
        fc1 = fc_layer(x, n_input, n_hidden_1, "fc1")
        relu = tf.nn.relu(fc1)
        tf.summary.histogram("fc3/relu", relu)
        fc2 = fc_layer(relu,n_hidden_1,n_hidden_2,"fc2")
        relu_2 = tf.nn.relu(fc2)
        tf.summary.histogram("fc3/relu", relu_2)
        logits = fc_layer(relu_2, n_hidden_2, n_output, "fc3")
    elif use_four_fc: 
        fc1 = fc_layer(x, n_input, n_hidden_1, "fc1")
        relu = tf.nn.relu(fc1)
        tf.summary.histogram("fc4/relu", relu)
        fc2 = fc_layer(relu,n_hidden_1,n_hidden_2,"fc2")
        relu_2 = tf.nn.relu(fc2)
        tf.summary.histogram("fc4/relu", relu_2)
        fc3 = fc_layer(relu_2,n_hidden_2,n_hidden_3,"fc3")
        relu_3 = tf.nn.relu(fc3)
        tf.summary.histogram("fc4/relu", relu_3)
        logits = fc_layer(relu_3, n_hidden_3, n_output, "fc4")
    elif use_five_fc: 
        fc1 = fc_layer(x, n_input, n_hidden_1, "fc1")
        relu = tf.nn.relu(fc1)
        tf.summary.histogram("fc5/relu", relu)
        fc2 = fc_layer(relu,n_hidden_1,n_hidden_2,"fc2")
        relu_2 = tf.nn.relu(fc2)
        tf.summary.histogram("fc5/relu", relu_2)
        fc3 = fc_layer(relu_2,n_hidden_2,n_hidden_3,"fc3")
        relu_3 = tf.nn.relu(fc3)
        tf.summary.histogram("fc5/relu", relu_3)
        fc4 = fc_layer(relu_3,n_hidden_3,n_hidden_4,"fc4")
        relu_4 = tf.nn.relu(fc4)
        tf.summary.histogram("fc5/relu", relu_4)
        logits = fc_layer(relu_4, n_hidden_4, n_output, "fc5")
    elif use_six_fc: 
        fc1 = fc_layer(x, n_input, n_hidden_1, "fc1")
        relu = tf.nn.relu(fc1)
        tf.summary.histogram("fc6/relu", relu)
        fc2 = fc_layer(relu,n_hidden_1,n_hidden_2,"fc2")
        relu_2 = tf.nn.relu(fc2)
        tf.summary.histogram("fc6/relu", relu_2)
        fc3 = fc_layer(relu_2,n_hidden_2,n_hidden_3,"fc3")
        relu_3 = tf.nn.relu(fc3)
        tf.summary.histogram("fc6/relu", relu_3)
        fc4 = fc_layer(relu_3,n_hidden_3,n_hidden_4,"fc4")
        relu_4 = tf.nn.relu(fc4)
        tf.summary.histogram("fc6/relu", relu_4)
        fc5 = fc_layer(relu_4,n_hidden_4,n_hidden_5,"fc5")
        relu_5 = tf.nn.relu(fc5)
        tf.summary.histogram("fc6/relu", relu_5)
        logits = fc_layer(relu_5, n_hidden_5, n_output, "fc6")
    elif use_seven_fc: 
        fc1 = fc_layer(x, n_input, n_hidden_1, "fc1")
        relu = tf.nn.relu(fc1)
        tf.summary.histogram("fc7/relu", relu)
        fc2 = fc_layer(relu,n_hidden_1,n_hidden_2,"fc2")
        relu_2 = tf.nn.relu(fc2)
        tf.summary.histogram("fc7/relu", relu_2)
        fc3 = fc_layer(relu_2,n_hidden_2,n_hidden_3,"fc3")
        relu_3 = tf.nn.relu(fc3)
        tf.summary.histogram("fc7/relu", relu_3)
        fc4 = fc_layer(relu_3,n_hidden_3,n_hidden_4,"fc4")
        relu_4 = tf.nn.relu(fc4)
        tf.summary.histogram("fc7/relu", relu_4)
        fc5 = fc_layer(relu_4,n_hidden_4,n_hidden_5,"fc5")
        relu_5 = tf.nn.relu(fc5)
        tf.summary.histogram("fc7/relu", relu_5)
        fc6 = fc_layer(relu_5,n_hidden_5,n_hidden_6,"fc6")
        relu_6 = tf.nn.relu(fc6)
        tf.summary.histogram("fc7/relu", relu_6)
        logits = fc_layer(relu_6, n_hidden_6, n_output, "fc7")
    
    else:
        logits = fc_layer(x, n_input, n_output, "fc")

    # Loss function 
    with tf.name_scope("xent"):
        # The positive and negative samples in the data are unbalanced. 
        # To push the algorithm to focus on fitting positives, I weighted the 
        # positive values more than the negative.
    
        maxY = tf.reduce_sum(y,1)* scale_class_weight
        class_weights = (maxY + 1)/6
        
        # Some expressions are more intense than others in the CK+ database and 
        # and that is weighted in the loss function by sample weights, sw. 
        # However, I got better results with just weighting all AUs
        # with equal intensity. 

#        mult_w = tf.multiply(y, sw) 
#        sum_w = tf.reduce_sum(mult_w,1)
#        
#        class_weights = ( sum_w + 1) / 6
        
        print(class_weights.get_shape())
        xent = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=y, name="xent"))
        xent = tf.reduce_mean(xent * class_weights)
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):
        zero = tf.constant(0, dtype=tf.float32)
    
        onesMat = tf.ones_like(logits)
        zerosMat = tf.zeros_like(logits)
        onesY = tf.ones_like(y,dtype=tf.float32)
        
        yFloat = tf.cast(y,dtype = tf.float32)    
        yFlipped = onesY - yFloat
        # PREDICTION - If logits >= 0, logits = 1, else logits = 0. 
        logitsBin = tf.cast(tf.where(logits>=zero,onesMat,zerosMat),dtype=tf.float32, name = "op_to_restore")
        
        
        tf.add_to_collection("coll", logitsBin)
        tf.add_to_collection("coll", x)
        
    
        print('logitsBin', logitsBin.get_shape())
        print('y', y.get_shape())
        print('where_logitsBin', tf.where(logitsBin)[:,1].get_shape())
        print('where_y', tf.where(y)[:,1].get_shape())
        time_steps = tf.cast(tf.shape(y)[0], dtype = 'int32')
        print(time_steps.get_shape())
    
        nFacs = tf.count_nonzero(y,1,dtype=tf.float32)
        onesFacs = tf.ones_like(nFacs)
        nFacs_Zeros = onesFacs*numFacs - nFacs
    
        nFacs = tf.where(tf.equal(nFacs,zero), onesFacs, nFacs)
        nFacs_Zeros = tf.where(tf.equal(nFacs_Zeros,zero), onesFacs, nFacs_Zeros)
    
        # Find TPR, TNR, FPR, FNR. 
        matrix_positive = tf.cast(tf.equal(logitsBin, y) & tf.equal(yFloat, tf.constant(1,dtype=tf.float32)),dtype=tf.float32)
        correct_pos = tf.reduce_sum(matrix_positive) / tf.reduce_sum(yFloat)
        tf.summary.scalar("TruePosRate", correct_pos)
        
        matrix_negative = tf.cast(tf.equal(logitsBin, y) & tf.equal(yFloat, zero),dtype = tf.float32)
        correct_neg = tf.reduce_sum(matrix_negative) / tf.reduce_sum(yFlipped)
        tf.summary.scalar("TrueNegRate", correct_neg)
        
        matrix_falsePos = tf.cast(tf.not_equal(logitsBin, y) & tf.equal(y, zero),dtype = tf.float32) #or yFlipped = 1
        falsePos = tf.reduce_sum(matrix_falsePos) / tf.reduce_sum(yFlipped)
        tf.summary.scalar("falsePosRate", falsePos)
        
        matrix_falseNeg = tf.cast(tf.not_equal(logitsBin, y) & tf.equal(yFloat, tf.constant(1,dtype=tf.float32)),dtype = tf.float32)
        falseNeg = tf.reduce_sum(matrix_falseNeg) / tf.reduce_sum(yFloat)
        tf.summary.scalar("falseNegRate", falseNeg)  
        
        
        tp_sum = tf.reduce_sum(matrix_positive,0)
        tp_sum_append = tf.concat([tf.constant([0],dtype = tf.float32), tp_sum],0)
        tf_sum = tf.reduce_sum(matrix_negative,0)
        fp_sum = tf.reduce_sum(matrix_falsePos,0)
        fn_sum = tf.reduce_sum(matrix_falseNeg,0)

        # Get Matrix of Confusion for multiclass binary classifier. 
        confusion = tf.Variable(initial_value = tf.zeros([n_output+1,n_output+1]), name = 'confusion')
        confusion1 = tf.Variable(initial_value = tf.cast( tf.diag(np.repeat(1,n_output+1)),dtype = tf.float32), name = 'confusion1')
        confusion2 = tf.Variable(initial_value = tf.zeros([n_output+1,n_output+1]), name = 'confusion2')
        confusion3 = tf.Variable(initial_value = tf.zeros([n_output+1,n_output+1]), name = 'confusion3')
        confusion4 = tf.Variable(initial_value = tf.zeros([n_output+1,n_output+1]), name = 'confusion4')

        confusion1 = confusion1[0,0].assign(5)
        confusion1= confusion1 * tp_sum_append
        confusion2 = confusion2[0,0].assign(tf.reduce_sum(tf_sum))
        confusion3 = tf.assign(confusion3[0,1:n_output+1],fp_sum)
        confusion4 = confusion4[1:n_output+1,0].assign(fn_sum)
        
        confusion = confusion1 + confusion2 + confusion3 +confusion4
    
        txtConfusion = tf.as_string(confusion,precision = 0, name='txtConfusion')
    
        tf.summary.text('txtConfusion', txtConfusion)
        
        correct_prediction = tf.cast(tf.equal(logitsBin,y),dtype = tf.float32, name = "correct_prediction")
            
        accuracy = tf.reduce_mean(correct_prediction, name = "accuracy")
        
        tf.summary.scalar("accuracy", accuracy)
    

# Summary for tensorboard
    summ = tf.summary.merge_all()

    saver = tf.train.Saver()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    
  
    writer = tf.summary.FileWriter(LOGDIR + hparam + '/train')
    test_writer = tf.summary.FileWriter(LOGDIR + hparam + '/test')
    writer.add_graph(sess.graph)


    for i in range(3001):
        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: train_x, y: train_y, sw:sw_train})
            sess.run([confusion],feed_dict={x: test_x, y: test_y, sw: sw_test})


            writer.add_summary(s, i)

        if i % 50 == 0:
            [acc,s] = sess.run([accuracy, summ],feed_dict={x: test_x, y: test_y, sw: sw_test})
            sess.run([confusion],feed_dict={x: test_x, y: test_y, sw:sw_test})
            test_writer.add_summary(s,i)
            saver.save(sess, os.path.join(savepath,hparam, "model"), i)
        sess.run(train_step, feed_dict={x: train_x, y: train_y, sw: sw_train})
    
def make_hparam_string(learning_rate, scale_class_weight,use_two_fc, use_three_fc, use_four_fc, use_five_fc, use_six_fc, use_seven_fc):
    fc_param = "fc=2" if use_two_fc else ""
    fc_param_3 = "fc=3" if use_three_fc else ""
    fc_param_4 = "fc=4" if use_four_fc else ""
    fc_param_5 = "fc=5" if use_five_fc else ""
    fc_param_6 = "fc=6" if use_six_fc else ""
    fc_param_7 = "fc=7" if use_seven_fc else ""
    return "lr_%.0E,cw_%i,%s%s%s%s%s%s" % (learning_rate, scale_class_weight, fc_param, fc_param_3, fc_param_4, fc_param_5, fc_param_6, fc_param_7)

def main():
  # Testing learning rates
    for learning_rate in [1E-2, 1E-3, 1E-4]:
        for scale_class_weight in [1]:

    # Include "False" as a value to try different model architectures
            for p in range(0,6):
                for use_two_fc in [p==0]:
                    for use_three_fc in [p==1]:
                        for use_four_fc in [p==2]:
                            for use_five_fc in [p==3]:
                                for use_six_fc in [p==4]:
                                    for use_seven_fc in [p==5]:
            # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2")
                                        hparam = make_hparam_string(learning_rate,scale_class_weight, use_two_fc, use_three_fc, use_four_fc, use_five_fc, use_six_fc, use_seven_fc)
                                        print('Starting run for %s' % hparam)
                                
                                        # Run
                                        facs_model(learning_rate, scale_class_weight,use_two_fc, use_three_fc, use_four_fc, use_five_fc, use_six_fc, use_seven_fc, hparam)
    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
    print('Running on mac? If you want to get rid of the dialogue asking to give '
    'network permissions to TensorBoard, you can provide this flag: '
    '--host=localhost')
        

if __name__ == '__main__':
  main()


