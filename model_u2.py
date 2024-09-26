from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras import regularizers

def create_binary_classification_model(input_shape):
    # Create a sequential model
    model = Sequential()

    # BLOCK 1 Add the first convolutional layer with L1 regularization, 32 filters, a 5x5 kernel, and 'relu' activation
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape,
                     kernel_regularizer=regularizers.l1(0.0001)))
    
    #model.add(BatchNormalization())

    model.add(Conv2D(32, (5, 5), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))

    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # BLOCK 2 Add a second convolutional layer with L1 regularization, 64 filters, a 3x3 kernel, and 'relu' activation
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_regularizer=regularizers.l1(0.0001)))
    
    #model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    

    # Flatten the output from the previous layer
    model.add(Flatten())

    # Add a fully connected layer with L2 regularization, 128 units, 'relu' activation, and dropout
    model.add(Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l1(0.0001)))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))

    # Add the output layer with 1 unit and sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
