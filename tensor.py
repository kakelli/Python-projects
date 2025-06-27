import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import MobileNetV2

#Creating a tensor constant
'''a = tf.constant(2)
b = tf.constant(3)

c = tf.add(a,b) #Adding two tensors
print(c)'''

#Detailed tensor
'''t = tf.constant([[1,2], [3,4]])
print(t.shape) #Gives the shape 
print(t.dtype)'''

#Tensorflow with numpy
'''a = tf.constant(np.array([1,2,3]), dtype=tf.float32)
print(tf.reduce_sum(a)) #Sum of all elements in the tensor'''

#Variables and Operations
'''x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    y = x*x
grad = tape.gradient(y,x)
print(grad.numpy())'''

#Eager Execution
'''print(tf.executing_eagerly()) #True in TF2'''

#Activation Functions
'''x = tf.constant([-1.0,0.0,1.0])
print(tf.nn.relu(x))'''

#Building a neural network(Keras)

model = Sequential([
    Dense(10, activation='relu', input_shape=(5,)),  # Input layer with 5 features
    Dense(1)
]
)

#Compile and Train
model.compile(optimizer='adam', loss='mean_squared_error')

x= np.random.rand(100,5)  # 100 samples, 5 features
y = np.random.rand(100, 1)  # 100 target values'''

'''model.fit(x,y, epochs=10)'''

#Custom training loop
'''optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()
for epochs in range(5):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))'''

#Custom Layers
'''class MyLayer(tf.keras.layers.Layer):
    def __init__(self,units):
        super().__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(shape = (input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w)'''

#Transfer Learning(Using pre-trained models)

'''base_model = MobileNetV2(weights='imagenet', include_top = False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze the base model'''

#Serving and Saving model
'''model.save('my_model.h5')
model =tf.keras.models.load_model('my_model.h5')'''

#Tensorflow for visualization
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "./logs")
model.fit(x,y, epochs = 5, callbacks=[tensorboard_callback])
