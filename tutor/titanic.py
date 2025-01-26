import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import clear_output

# Load dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# Ensure categorical columns are treated as strings
for feature_name in CATEGORICAL_COLUMNS:
    dftrain[feature_name] = dftrain[feature_name].astype(str)
    dfeval[feature_name] = dfeval[feature_name].astype(str)

# Create feature preprocessing layers
feature_columns = []

# Categorical columns: StringLookup expects input to be strings
for feature_name in CATEGORICAL_COLUMNS:
    feature_columns.append(
        layers.StringLookup(vocabulary=dftrain[feature_name].unique(), output_mode='int')
    )

# Numeric columns: Apply normalization to numeric columns
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(
        layers.Normalization()
    )

# Define the input layers for categorical and numeric columns
inputs = {feature_name: layers.Input(name=feature_name, shape=(), dtype=tf.string)
          for feature_name in CATEGORICAL_COLUMNS}

for feature_name in NUMERIC_COLUMNS:
    inputs[feature_name] = layers.Input(name=feature_name, shape=(1,), dtype=tf.float32)

# Apply feature preprocessing layers
encoded_features = []

# Apply StringLookup for categorical features
for feature_name in CATEGORICAL_COLUMNS:
    encoded = layers.StringLookup(vocabulary=dftrain[feature_name].unique(), output_mode='int')(inputs[feature_name])
    encoded_features.append(layers.Reshape((1,))(encoded))  # Ensure the shape is (None, 1)

# Apply Normalization for numeric features
for feature_name in NUMERIC_COLUMNS:
    normalized = layers.Normalization()(inputs[feature_name])
    encoded_features.append(normalized)

# Concatenate all features
all_features = layers.concatenate(encoded_features)

# Build the Keras model
x = layers.Dense(128, activation='relu')(all_features)
output = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Prepare input functions
def df_to_dataset(data_df, label_df, shuffle=True, batch_size=32):
    # Convert DataFrame into dictionary format
    data_dict = {name: data_df[name] for name in data_df.columns}
    ds = tf.data.Dataset.from_tensor_slices((data_dict, label_df))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data_df))
    ds = ds.batch(batch_size)
    return ds

train_ds = df_to_dataset(dftrain, y_train)
eval_ds = df_to_dataset(dfeval, y_eval, shuffle=False)

# Train the model
model.fit(train_ds, validation_data=eval_ds, epochs=10)

# Evaluate the model
result = model.evaluate(eval_ds)
clear_output()
print(result)
