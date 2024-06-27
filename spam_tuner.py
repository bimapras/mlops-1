import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
import keras_tuner as kt

LABEL_KEY = "label"
FEATURE_KEY = "email"

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy",
    mode="max",
    verbose=1,
    patience=2,
    baseline=1.0
)

def transformed_name(key):
    """Rename transformed key"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs,
             batch_size=32) -> tf.data.Dataset:
    """Get post_transform feature & create batches of data"""
    
    # Get post_transform feature spec
    transform_feature_spec = (tf_transform_output.transformed_feature_spec().copy())
    
    # Create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset

# Vocabulary size and number of words in a sequence.
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

def model_builder(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), dtype=tf.string, name=transformed_name(FEATURE_KEY)),
        tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1])),
        vectorize_layer,
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=hp.Int("embed_dims", min_value=16, max_value=64, step=32), name="embedding"),
        tf.keras.layers.LSTM(units=hp.Int('lstm_units',min_value=32, max_value=64, step=32), return_sequences=True),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(units=hp.Int('dense_units', min_value=32, max_value=64, step=32), activation='relu'),
        tf.keras.layers.Dropout(rate=hp.Float('drop_rate', min_value=0.1, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    tf.keras.backend.clear_session()
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    
    return model

def tuner_fn(fn_args):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)
    
    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)]])
    
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_binary_accuracy',
        max_trials=20,
        directory=fn_args.working_dir,
        project_name='spam_classification'
    )
    
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [early_stopping_callback],
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 10
        },
    )
