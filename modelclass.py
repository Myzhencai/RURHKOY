import tensorflow as tf
from layerClass import Layer

class Model(Layer,):

    def __init__(self, inputPlaceholder, outputPlaceHolder,isTrainPlaceHolder,learningRate):
        print ("Initialisation")
        self.initializer = "Xavier"
        Layer.__init__(self,self.initializer)
        self.input = inputPlaceholder
        self.output = outputPlaceHolder
        self.learningRate = learningRate
        self.isTrain = isTrainPlaceHolder
        self._prediction = None
        self._optimize = None
        self._loss = None

    def prediction(self):
        print ("Prediction")
        if not self._prediction:
            self._prediction = Layer.runBlock(self, inputData = self.input, isTrain = self.isTrain)
        return self._prediction

    def error(self):
        with tf.compat.v1.name_scope('loss'):
            if not self._loss:
                numerator = 2 * tf.reduce_sum(input_tensor=self.output * self._prediction, axis=(1,2,3))
                denominator = tf.reduce_sum(input_tensor=self.output*self.output + self._prediction*self._prediction+0.00000006, axis=(1,2,3))
                self.dice = tf.reshape(1 - (numerator / denominator), (-1, 1, 1, 1))
                print("dice", self.dice.shape)
                self.loss1 = tf.nn.weighted_cross_entropy_with_logits(labels=self.output, logits=self._prediction,
                                                                      pos_weight=-0.2)
                self.loss2 = tf.reduce_sum(input_tensor=(self.output-self._prediction)*(self.output-self._prediction))
                self._loss = tf.reduce_mean(input_tensor=self.loss1 + self.dice+self.loss2)
                # self._loss = tf.reduce_mean(input_tensor=self.loss1 + self.dice)
                ##################################################################################################################
                #imY = tf.cast(self.output,tf.bool)
                #imlogits = tf.cast(self._prediction, tf.bool)
                #intersection = tf.math.logical_and(imY,imlogits)

                #self.dice = 2. * tf.reduce_sum(tf.cast(intersection,tf.float32)) / (tf.reduce_sum(tf.cast(imY,tf.float32)) + tf.reduce_sum(tf.cast(imlogits,tf.float32)))
                #loss = tf.losses.sigmoid_cross_entropy(Y, logits) + dic
                #####################################################################################################
                # numerator = tf.reduce_sum(2*tf.multiply(self.output, self._prediction))
                # denominator = tf.reduce_sum(
                #     tf.multiply(self.output , self.output )+ tf.multiply(self._prediction , self._prediction) + 0.00000006)
                # self.dice=1 - numerator / denominator
                # self.loss1 = -0.2 * tf.reduce_sum(tf.multiply(self.output, tf.math.log(self._prediction[0])))
                # self._loss = tf.reduce_mean(self.loss1 + self.dice)
                #########################################################################################################

            return self._loss


    def optimize(self):
        with tf.compat.v1.name_scope('optimiser'):
            if not self._optimize:
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learningRate)
                self._optimize = optimizer.minimize(self._loss)
            return self._optimize




            
