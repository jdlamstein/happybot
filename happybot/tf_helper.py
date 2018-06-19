# -*- coding: utf-8 -*-
"""
Module for loading tensorflow model and analyzing results. 
"""

import tensorflow as tf
import numpy as np

class ImportGraph():
    """Import and run isolated TF graph.
    
    Args:
        loc (str): Path to TF graph.
        
    Attributes:
        graph: TF graph.
        sess: TF session.
        activation: Get collection of TF saved variables.
        
    Use:
        data = 50         # random data
        model = ImportGraph('models/model_name')
        result = model.run(data)
        print(result)
    """
        
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
              # FROM SAVED COLLECTION:            
            self.activation = tf.get_collection('coll')[0]

              # BY NAME:
#            self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    def run(self, data):
        """Run the activation operation previously imported.
        
        Returns:
            Array of predictions of facial actions units."""
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.activation, feed_dict={"x:0": data})
        
class facs2emotion():
    """Predict basic emotion based on facial action units.
    
    Args:
        facsT (int): Predicted upper facial action units.
        facsB (int): Predicted lower facial action units.
        
    Attributes:
        facsT (int): Predicted upper facial action units.
        facsB (int): Predicted lower facial action units.
        emo (int): Predicted basic emotion. 
    """

    def __init__(self, facsT, facsB):
        self.facsT = facsT
        self.facsB = facsB
        self.emo = 0
    
    def declare(self):
        """Predict basic emotion using decision tree method."""
        # Use numpy arrays.
        T = np.array(self.facsT)
        B = np.array(self.facsB)
        # Get indices of arrays. 
        idxT = np.where(T==1)
        idxB = np.where(B==1)
        idxT = np.array(idxT)
        idxB = np.array(idxB)
        if (4 in idxT and 2 in idxB):
            self.emo = 5 # happiness
        elif (0 in idxT and 2 in idxT and 3 in idxB): # AU 9 (idx 1) for AU 10 (idx 2)
            self.emo = 6 # sadness
        elif (0 in idxT and 1 in idxT and 3 in idxT and 8 in idxB ): #exchanging jaw drop AU 26 for lips part 25
            self.emo = 7 # surprise
        elif (0 in idxB and (2 in idxT or 5 in idxT or 4 in idxB ) ):
            self.emo = 3 # disgust (missing FAU 16)
        elif (5 in idxB and (0 in idxT or 2 in idxT or 3 in idxT) ):  #exchanging jaw drop AU 26 for lips part 25
            self.emo = 4 # fear
        elif (6 in idxB and (4 in idxB or 7 in idxB) and (2 in idxT or 5 in idxT)): 
            self.emo = 1 # anger
        elif (2 in idxB and 4 in idxB):
            self.emo = 2 # contempt (missing FAU 14)
        else:
            self.emo = 0
        
        return self.emo
     