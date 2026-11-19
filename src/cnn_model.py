# Author: Andres Rosso (CS 4371 Group 1)

# Overall Architecture Of This 1D Convulsional Neural Network (CNN):
# Step 1: Setting Seeds                             -- So it can be reproduced with ease (Neural Networks are built on randomness of weights and 
#                                                      shuffling of data): This seed "locks" it.

# Step 2: Reshaping The Data                        -- CNN's were oringally 3 dimensional (For Images) but since we are feeding it 1 dimensional
#                                                      stream of data we need to reshape it to 3D as tensorflow's Conv1D demands a 3D Grid like a 
#                                                      image. So, we take our shape of our data (6000, 45) and reshape it to (6000, 45, 1).
#                                                      for images: (Height, Width, 3(RGB 3 channels)) where (Samples, Features, Channels) we only 
#                                                      use 1 channel in our model because our only channel is network traffic

# Step 3: Splitting The Training Data               -- We split the training data 80/20 letting the model train on 80% and test itself on the 
#                                                      remaining 20% of it because we dont want to feed it all the data and have it memorize 
#                                                      answers (Overfitting) it.

# Step 4: Building Actual CNN Architecture: (Based off Paper) (Conv1D -> Pooling -> Conv1D -> Pooling -> Dense)
#       Sub Step 1: Conv1D (32 filters, size 3)     -- Slides a window of size 3 across our 45 features, to find local patterns 
#                                                      (for example: if 3 out of 4 spike together it looks SUS 🔪ඞ Imposter) 
#                                                      (Turns it into 2D (Sample, Channels): (45, 1) --> (45, 32)) but remember this 2D array is
#                                                      only for a single batch but the overall (6000, 45, 1) is still a 3D array. We do this 
#                                                      because we are processing this in batches of 32.

#       Sub Step 2: MaxPooling1D (size 2)           -- To prevent overfitting (memorization of data) we cut the data size in half keeping only 
#                                                      the strongest signals. (Stays 2D just now: (45, 32) --> (22, 32))

#       Sub Step 3: Conv1D (64 filters, size 3)     -- Does the same as Sub Step 1 except has more filters: 64 vs 32 to find more patterns based
#                                                      off the first layer's findings (Stays 2D: (22, 32) --> (22, 64))

#       Sub Step 4: MaxPooling1D (size 2)           -- Again we prevent overfitting by chopping the data size in half again keeping only strongest
#                                                      signals. (Stays 2D: (22, 64) --> (11, 64)).

#       Sub Step 5: Flatten                         -- We take this 2D array (11, 64) and flatten it to a 1D array 
#                                                      (think of it like taking a 11 x 64 grid/matrix and unrolling it into a single line so 
#                                                       11 * 64 = 704 so we get: (11, 64) --> (704) ).

#       Sub Step 6: Dense (128 neurons, ReLU)       -- We flattened it to 1D before we got to Dense step because the Dense layer takes a 1D array
#                                                      as input and we had a 2D array.

#       Sub Step 7:                                 -- The paper uses Dense(num_classes, activation='softmax') because it makes a guess of which 
#                                                      of the 18 attack vectors it is but we will use Dense(1, activation='sigmoid') because we 
#                                                      only care if it is malicious or not.

# Step 5: Training The Model (Batches & Epochs)     -- We need to pass through our data to the model but we do so in "chunks" called "Batches" of 
#                                                      32 rows because if we feed it all 6000 rows of data at once we might overload and crash 
#                                                      our memory (due to low RAM 😭crying ). This way the model calulates its error and moves on to the
#                                                      Next 32 rows and so on until until it completes all 6000 rows. When it completely finishes
#                                                      Those 6000 rows it is called a "Epoch" (a full pass of the data into the model). Now 1 
#                                                      epoch is not enough but 1000 epochs would cause our model to memorize the data which is
#                                                      overfitting it so here we will do 10 epochs to see the loss drop and accuracy rise.

# Step 6: Making predictions & Thresholding        -- We will test our model on the data inside the "test" folder inside the "data" folder then the
#                                                     model will produce/spit out a probability like (0.76) for every single row and the threshold
#                                                     is the place where we draw line on the sand that makes the output we produce is either 1(true)
#                                                     or 0(false). We can set that threshold to be like (x >= 0.85) = 1 or anything else.

# Step 7: Save Results                             -- We will first take those array of predictions and probabilites and save them into 2 .npy files
#                                                     and save them where the "results" folder is.

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from src.data_loader import load_data

# Step 1: Set Seed/ other const data
SEED = 42
THRESHOLD = 0.75
EPOCHS = 10 
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Step 2: Reshaping the data
def reshape_for_cnn(X):
    return X.reshape(-1, X.shape[1], 1)          # Reshapes the data to 3D

# Step 4: Building Actual CNN Architecture: (Based off Paper) (Conv1D -> Pooling -> Conv1D -> Pooling -> Dense): 
# If you notice we skipped step 3 that is because in step 3 we will need to call load the model so we build the model before.
def build_model_cnn(input_shape):
    """Create and compile the CNN model."""
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # All we changed from paper was to sigmoid activation because softmax makes all calculated 
                                               # probabilites add up to 1 and forces it to choose a winner like 85% DDOS 10% Begign 5% SPOOF
                                               # it would be forced to choose spoofing whereas sigmoid only chooses 1 or 0 being if its a attack
                                               # or not which is binary we dont care what exact attack it is.

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # We also changed from loss = catagorical_crossentropy to 
                                                                                      # binary because we only need to know if malicious or not
    return model

# This function will contain code for steps 3 and 5.
def train_model(X_train, Y_train):
    X_train_cnn = reshape_for_cnn(X_train) # We reshape here because .fit() requires a 3D array.

    # Step 3: Splitting the Training Data
    # Splits all data Across 45 samples 80/20 and stratify ensures that the spilt is even between test(80%) and validiation(20%) data
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_cnn, Y_train, test_size = 0.20, random_state = SEED, stratify = Y_train)
    
    # Step 4: we need to build the CNN model above
    model = build_model_cnn((X_train.shape[1], 1))

    # Step 5: Training The Model (Batches & Epochs):
    # We first take that 80% (6000 rows * 0.8 = 4800 rows) training data which we split and train the model in batches of 32 rows and iterate
    # through this entire dataset 10 times or 10 epochs using this built in Karras function .fit() which handles the training, and evaluates 
    # its performance after every cycle using our separate validation data (the 20% that was split).
    model.fit(X_tr, y_tr, validation_data = (X_val, y_val), epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 2)
    return model

# Step 6: Making predictions & Thresholding (we set the threshold eariler) (IMPORTANT: Threshold determines when model classiflies it as malicious)
def predictions(model, X_test):
    reshaped_X_test = reshape_for_cnn(X_test)
    probabilities = model.predict(reshaped_X_test).ravel() # calculates the probability of detecting a attack and .ravel() flattens result to 1D array
    binary_prediction = (probabilities >= THRESHOLD).astype(int) # The line which outputs 1 or 0 determining whether the model thinks its malicious or not
    score = probabilities
    y_pred = binary_prediction
    return y_pred, score

# Step 7: Save Data to Results/ Main
if __name__ == "__main__":
    # Load data, train model, make predictions/probabilities on test data
    data = load_data()
    model = train_model(data["X_train"], data["y_train"])
    y_pred, score = predictions(model, data["X_test"])

    # verify outputs
    print("\n--- HANDOFF CHECKS ---")
    print("Test rows:", len(data["X_test"]))
    print("Prediction count:", len(y_pred))
    print("Probabilities range:", score.min(), "to", score.max())
    print("Unique predictions:", sorted(set(y_pred.tolist())))

    # save the results in results folder (the 2 .npy files)
    src_dir = os.path.dirname(__file__) 
    root_dir = os.path.dirname(src_dir) 
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "cnn_predictions.npy"), y_pred)
    np.save(os.path.join(results_dir, "cnn_probabilities.npy"), score)


