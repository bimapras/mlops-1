import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
import os
import json
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "label"
FEATURE_KEY = "email"


def transformed_name(key):
    """Rename transformed key"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs,
             batch_size=32)->tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""
    
    # Get post_transform feature spec
    transform_feature_spec = (tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = transformed_name(LABEL_KEY))
    return dataset
 
# Vocabulary size and number of words in a sequence.
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100
 
vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)
 
 
embedding_dim=16
def model_builder(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), dtype=tf.string, name=transformed_name(FEATURE_KEY)),
        tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1])),
        vectorize_layer,
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=hp['embed_dims'], name="embedding"),
        tf.keras.layers.LSTM(hp['lstm_units'], return_sequences=True),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(hp['dense_units'], activation='relu'),
        tf.keras.layers.Dropout(hp['drop_rate']),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss = 'binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate']),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    
    )
    
    # print(model)
    model.summary()
    return model 
 
 
def _get_serve_tf_examples_fn(model, tf_transform_output):
    
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        
        feature_spec = tf_transform_output.raw_feature_spec()
        
        feature_spec.pop(LABEL_KEY)
        
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = model.tft_layer(parsed_features)
        
        return model(transformed_features)
        
    return serve_tf_examples_fn
    
def run_fn(fn_args: FnArgs) -> None:
    hp = fn_args.hyperparameters["values"]
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir, update_freq='batch'
    )
    
    # Create callback early stopping and model checkpoint
    es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=2)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)
    
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)
    
    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)]])
    
    # Build the model
    model = model_builder(hp)
    
    # Train the model
    model.fit(x = train_set,
            validation_data = val_set,
            callbacks = [tensorboard_callback, es, mc],
            epochs=10)
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples'))
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
