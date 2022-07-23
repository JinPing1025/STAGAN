
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU 


def gumbel_softmax(x,eps=1e-20,delta=5,axis=1,hard=False):
    shape=tf.shape(x)[1:]
    a = tf.random.uniform(shape,minval=0,maxval=1)
    b = -tf.math.log(-tf.math.log(a+eps) + eps)
    gumbel_perturbed_logp =  x + b
    y = tf.nn.softmax(gumbel_perturbed_logp/delta, axis=axis)
    if hard:
        k=tf.shape(x)[-1]
        y_hard=tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    else:
        y = y
    return y 

def GraphGenerator(latent_dim, adjacency_shape, feature_shape,):
    z = keras.layers.Input(shape=(latent_dim,))
    x = keras.layers.BatchNormalization()(z)
    # x = z
    x1 = keras.layers.Dense(128)(x)
    x1 = keras.layers.BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    x2 = keras.layers.Dense(256)(x1)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)
    x2 = keras.layers.Dropout(0.2)(x2)
    x3 = keras.layers.Dense(512)(x1)
    x3 = keras.layers.BatchNormalization()(x3)
    x3 = LeakyReLU()(x3)
    x3 = keras.layers.Dropout(0.2)(x3)
    x4 = keras.layers.Concatenate()([x2,x3])

    x_adjacency = keras.layers.Dense(tf.math.reduce_prod(adjacency_shape))(x4)
    x_adjacency = keras.layers.Reshape(adjacency_shape)(x_adjacency)
    x_adjacency = (x_adjacency + tf.transpose(x_adjacency, (0, 1, 3, 2))) / 2
    # x_adjacency = keras.layers.Softmax(axis=1)(x_adjacency)
    x_adjacency = gumbel_softmax(x_adjacency,eps=1e-20,delta=5,axis=1)

    x_features = keras.layers.Dense(tf.math.reduce_prod(feature_shape))(x4)
    x_features = keras.layers.Reshape(feature_shape)(x_features)
    # x_features = keras.layers.Softmax(axis=2)(x_features)
    x_features = gumbel_softmax(x_features,eps=1e-20,delta=5,axis=2)
    return keras.Model(inputs=z, outputs=[x_adjacency, x_features], name="Generator")


class GraphConvlayer(keras.layers.Layer):
    def __init__(self,units=128,activation='relu',use_bias=False,kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,**kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
    
    def build(self,input_shape):
        bond_dim = input_shape[0][1]
        atom_dim = input_shape[1][2]
        self.W = self.add_weight('W',shape=(bond_dim,atom_dim,self.units),initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,trainable=True,dtype=tf.float32)
        if self.use_bias:
            self.bias = self.add_weight('bias',shape=(bond_dim,1,self.units),initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,trainable=True,dtype=tf.float32)
        self.build = True
    
    def call(self,inputs,training=False):
        adjacency, features = inputs
        x = tf.matmul(adjacency,features[:, None, :, :])  #(None,5,13,13)*(None,1,13,5)=(None,5,13,5)
        #x = tf.add(x,features[:, None, :, :])
        x = tf.matmul(x, self.W)     #(None,5,13,5)*(None,5,5,128)=(None,5,13,128)
        if self.use_bias:
            x += self.bias
        x_reduced = tf.reduce_sum(x, axis=1)   #(None,13,128)
        x_reduced = self.activation(x_reduced)
        return x_reduced
        

def GraphDiscriminator(adjacency_shape, feature_shape):
    adjacency = keras.layers.Input(shape=adjacency_shape)
    features = keras.layers.Input(shape=feature_shape)
    features_transformed = features
    features_transformed = GraphConvlayer(64)([adjacency, features_transformed])
    features_transformed = GraphConvlayer(128)([adjacency, features_transformed])
    features_transformed = GraphConvlayer(256)([adjacency, features_transformed])
    features_transformed = GraphConvlayer(256)([adjacency, features_transformed])
    
    x = keras.layers.GlobalAveragePooling1D()(features_transformed)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x_out = keras.layers.Dense(1, dtype="float32")(x)
    return keras.Model(inputs=[adjacency, features], outputs=x_out)


class GraphDRAGAN(keras.Model):
    def __init__(self,generator,discriminator,discriminator_steps=1,generator_steps=1,gp_weight=10,**kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps
        self.gp_weight = gp_weight
        self.latent_dim = self.generator.input_shape[-1]
        
    def compile(self, optimizer_generator, optimizer_discriminator, **kwargs):
        super().compile(**kwargs)
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.metric_generator = keras.metrics.Mean(name="loss_gen")
        self.metric_discriminator = keras.metrics.Mean(name="loss_dis")
        
    def train_step(self, inputs):
        if isinstance(inputs[0], tuple):
            inputs = inputs[0]        
        graph_real = inputs
        self.batch_size = tf.shape(inputs[0])[0]
        for step in range(self.discriminator_steps):
            z = tf.random.normal((self.batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                graph_generated = self.generator(z, training=True)
                loss = self.loss_discriminator(graph_real, graph_generated)
            grads = tape.gradient(loss, self.discriminator.trainable_weights)
            self.optimizer_discriminator.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            self.metric_discriminator.update_state(loss)
        
        for step in range(self.generator_steps):
            z = tf.random.normal((self.batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                graph_generated = self.generator(z, training=True)
                loss = self.loss_generator(graph_generated)
            grads = tape.gradient(loss, self.generator.trainable_weights)
            self.optimizer_generator.apply_gradients(zip(grads, self.generator.trainable_weights))
            self.metric_generator.update_state(loss) 
        return {m.name: m.result() for m in self.metrics}
        
    
    def loss_discriminator(self,graph_real,graph_generated):
        logits_real = self.discriminator(graph_real, training=True)
        logits_generated = self.discriminator(graph_generated, training=True)
        loss = tf.reduce_mean(logits_generated) - tf.reduce_mean(logits_real)
        loss_gp = self.gradient_penalty(graph_real)
        return loss + loss_gp * self.gp_weight

    def loss_generator(self,graph_generated):
        logits_generated = self.discriminator(graph_generated, training=True)
        return -tf.reduce_mean(logits_generated)

    def gradient_penalty(self,graph_real):
        adjacency_real, features_real = graph_real
        def interpolate(a,ndims):
            beta = tf.random.uniform(tf.shape(a), 0., 1.)
            b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (ndims - 1)
            alpha = tf.random.uniform(shape, 0., 1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter
        
        adjacency_interp = interpolate(adjacency_real,4)
        features_interp = interpolate(features_real,3)
        with tf.GradientTape() as tape:
            tape.watch(adjacency_interp)
            tape.watch(features_interp)
            logits = self.discriminator([adjacency_interp, features_interp], training=True)
        grads = tape.gradient(logits, [adjacency_interp, features_interp])
        grads_adjacency_penalty = (1 - tf.norm(grads[0], axis=1)) ** 2
        grads_features_penalty = (1 - tf.norm(grads[1], axis=2)) ** 2
        return tf.reduce_mean(tf.reduce_mean(grads_adjacency_penalty, axis=(-2, -1)) + 
                              tf.reduce_mean(grads_features_penalty, axis=(-1)))
    
    