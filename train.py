# training
# updated training to increase complexity of regression models, switched to mean absolute error, lowered learning rate
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau

import time

start_time = time.perf_counter()

# Load dataset from .npz file
def load_dataset(npz_file):
    data = np.load(npz_file)
    return (data['top_view_images'], data['side_view_images'], data['depth_data'], 
            data['cross_section_25'], data['cross_section_75'], data['performance_data'])

def build_full_model(input_shape=(128, 128, 1), reg_input_shape=(1,), cross_section_shape=(600,)):
    # Image input 1 (Top view)
    input1 = Input(shape=input_shape, name='Top_View_Input')
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    
    #added by SR
    x1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    
    x1 = GlobalAveragePooling2D()(x1)

    # Image input 2 (Side view)
    input2 = Input(shape=input_shape, name='Side_View_Input')
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(input2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    
    #added by SR
    x2 = Conv2D(512, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    
    x2 = GlobalAveragePooling2D()(x2)

    # Regression input (depth data)
    reg_input = Input(shape=reg_input_shape, name='Regression_Input')
    reg_branch = Dense(128, activation='relu')(reg_input)  # Increased number of perceptrons
    reg_branch = Dense(256, activation='relu')(reg_branch)
    reg_branch = Dense(512, activation='relu')(reg_branch)  # Increased number of perceptrons
    reg_branch_output = Dense(128, activation='relu')(reg_branch)

    # Cross-section inputs
    cross_input1 = Input(shape=cross_section_shape, name='Cross_Section_Input1')
    cross_branch1 = Dense(128, activation='relu')(cross_input1)  # Increased number of perceptrons
    cross_branch1 = Dense(256, activation='relu')(cross_branch1)
    cross_branch1 = Dense(512, activation='relu')(cross_branch1)  # Increased number of perceptrons
    
    #added by SR
    cross_branch1 = Dense(1024, activation='relu')(cross_branch1)
    cross_branch1 = Dense(512, activation='relu')(cross_branch1)
    
    cross_branch_output1 = Dense(128, activation='relu')(cross_branch1)

    cross_input2 = Input(shape=cross_section_shape, name='Cross_Section_Input2')
    cross_branch2 = Dense(128, activation='relu')(cross_input2)  # Increased number of perceptrons
    cross_branch2 = Dense(256, activation='relu')(cross_branch2)
    cross_branch2 = Dense(512, activation='relu')(cross_branch2)  # Increased number of perceptrons
    
    #added by SR
    cross_branch2 = Dense(1024, activation='relu')(cross_branch2)
    cross_branch2 = Dense(512, activation='relu')(cross_branch2)
    
    cross_branch_output2 = Dense(128, activation='relu')(cross_branch2)

    # Concatenate all branches
    concatenated = Concatenate()([x1, x2, reg_branch_output, cross_branch_output1, cross_branch_output2])

    # Fully connected layers after concatenation
    # Adding L2 regularization to dense layers
    
    #added by SR
    dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(concatenated)
    dense = Dropout(0.6)(dense)
    
    dense = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(concatenated)
    dense = Dropout(0.6)(dense)
    dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense)
    dense = Dropout(0.4)(dense)
    dense = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(dense)
    dense = Dropout(0.3)(dense)
    dense = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(dense)

    # Output layer for predicting Stress, Mass, Deflection
    output = Dense(3, name='Performance_Output')(dense)

    # Define the model
    model = Model(inputs=[input1, input2, reg_input, cross_input1, cross_input2], outputs=output)

    # Compile model with a lower starting learning rate
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='mae')  # Starting with a lower LR

    return model

# Main execution
npz_file = r'.\training_dataset.npz'
X1, X2, X_reg, X_cross1, X_cross2, y = load_dataset(npz_file)

# Split data into training and validation sets
X1_train, X1_val, X2_train, X2_val, X_reg_train, X_reg_val, X_cross1_train, X_cross1_val, X_cross2_train, X_cross2_val, y_train, y_val = train_test_split(
    X1, X2, X_reg, X_cross1, X_cross2, y, test_size=0.2, random_state=42
)

# Build and train the model with callbacks for early stopping and model checkpoint
model = build_full_model(input_shape=(128, 128, 1), reg_input_shape=(1,), cross_section_shape=(600,))
# Setting EarlyStopping with increased patience
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
# Adding ReduceLROnPlateau to reduce learning rate when validation loss stops improving
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, min_lr=1e-7)

history = model.fit(
    [X1_train, X2_train, X_reg_train, X_cross1_train, X_cross2_train], y_train,
    validation_data=([X1_val, X2_val, X_reg_val, X_cross1_val, X_cross2_val], y_val),
    epochs=250,  # Increase epochs to give the model more time to learn
    batch_size=16,  # Use smaller batch sizes for potentially more precise updates
    #callbacks=[early_stopping, model_checkpoint, reduce_lr_on_plateau]
)

end_time = time.perf_counter()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time/60} minutes")

# Plotting the loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Save the final trained model
model.save(r'all_inputs_more_layers4.h5')