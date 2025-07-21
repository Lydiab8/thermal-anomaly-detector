import tensorflow as tf

def build_autoencoder(input_shape):
    input_img = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input_img)
    encoded = tf.keras.layers.Dense(64, activation='relu')(x)
    decoded = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid')(encoded)
    decoded = tf.keras.layers.Reshape(input_shape)(decoded)

    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
