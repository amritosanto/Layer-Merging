"""Official TFDS fallback. Lazy imports leave TensorFlow out of normal HF runs."""
def tfds_module():
    try:
        import tensorflow as tf
        # This loader must not reserve either training GPU.
        tf.config.set_visible_devices([], 'GPU')
        import tensorflow_datasets as tfds
    except ImportError as error:
        raise RuntimeError('Official TFDS fallback requires tensorflow-datasets==4.9.8 and TensorFlow. '
                           'Kaggle normally provides TensorFlow; install TFDS explicitly.') from error
    return tfds

def configs():
    return sorted(tfds_module().builder_cls('wiki40b').builder_configs)

def rows(language,seed,buffer):
    tfds=tfds_module()
    builder=tfds.builder(f'wiki40b/{language}:1.3.0')
    builder.download_and_prepare()
    dataset=builder.as_dataset(split='train',shuffle_files=False)
    dataset=dataset.shuffle(buffer,seed=seed,reshuffle_each_iteration=False)
    for row in tfds.as_numpy(dataset):
        yield {k:v.decode('utf-8') if isinstance(v,bytes) else v for k,v in row.items()}
