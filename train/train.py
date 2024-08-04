# %% [markdown]
# ## Import Dependencies

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T08:08:57.976258Z","iopub.execute_input":"2024-08-02T08:08:57.977278Z","iopub.status.idle":"2024-08-02T08:09:00.836644Z","shell.execute_reply.started":"2024-08-02T08:08:57.977234Z","shell.execute_reply":"2024-08-02T08:09:00.834927Z"}}
#!pip freeze > requirements.txt

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T15:08:37.545084Z","iopub.execute_input":"2024-08-02T15:08:37.546319Z","iopub.status.idle":"2024-08-02T15:08:52.391252Z","shell.execute_reply.started":"2024-08-02T15:08:37.546270Z","shell.execute_reply":"2024-08-02T15:08:52.389926Z"}}
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf

from pathlib import Path
import tensorflow as tf

from keras import ops
from keras.models import Model
from keras.layers import *
import keras

target_shape = (105, 105)
epochs = 50

# %% [markdown]
# ## GPU

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:11:27.943673Z","iopub.execute_input":"2024-08-02T06:11:27.944077Z","iopub.status.idle":"2024-08-02T06:11:27.950478Z","shell.execute_reply.started":"2024-08-02T06:11:27.944046Z","shell.execute_reply":"2024-08-02T06:11:27.949326Z"}}
print(f'List all device available : {list(tf.config.list_physical_devices()[0])}')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print("Name:", gpu.name, "  Type:", gpu.device_type)

# %% [markdown]
# ## Load Image Path

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:11:30.679381Z","iopub.execute_input":"2024-08-02T06:11:30.680416Z","iopub.status.idle":"2024-08-02T06:11:30.684561Z","shell.execute_reply.started":"2024-08-02T06:11:30.680380Z","shell.execute_reply":"2024-08-02T06:11:30.683563Z"}}
anchor_images_path = '/kaggle/input/dataset-piko-face/Dataset Piko Face/anchor'
positive_images_path = '/kaggle/input/dataset-piko-face/Dataset Piko Face/positive'
negative_images_path = '/kaggle/input/dataset-piko-face/Dataset Piko Face/negative'

# %% [markdown]
# ### map image path

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:11:32.749201Z","iopub.execute_input":"2024-08-02T06:11:32.749604Z","iopub.status.idle":"2024-08-02T06:11:33.475007Z","shell.execute_reply.started":"2024-08-02T06:11:32.749567Z","shell.execute_reply":"2024-08-02T06:11:33.473797Z"}}
anchor_images = sorted([ os.path.join(anchor_images_path, f) for f in os.listdir(anchor_images_path)])
positive_images = sorted([ os.path.join(positive_images_path, f) for f in os.listdir(positive_images_path)])
negative_images = sorted([ os.path.join(negative_images_path, f) for f in os.listdir(negative_images_path)])

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:11:36.328434Z","iopub.execute_input":"2024-08-02T06:11:36.328823Z","iopub.status.idle":"2024-08-02T06:11:36.336946Z","shell.execute_reply.started":"2024-08-02T06:11:36.328790Z","shell.execute_reply":"2024-08-02T06:11:36.335646Z"}}
len(anchor_images), len(positive_images), len(negative_images)

# %% [markdown]
# ### load image path to dataset Object

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:11:38.288391Z","iopub.execute_input":"2024-08-02T06:11:38.288771Z","iopub.status.idle":"2024-08-02T06:11:38.344614Z","shell.execute_reply.started":"2024-08-02T06:11:38.288743Z","shell.execute_reply":"2024-08-02T06:11:38.343552Z"}}
anchor_dataset_path = tf.data.Dataset.from_tensor_slices(anchor_images).take(300).shuffle(len(anchor_images))
positive_dataset_path = tf.data.Dataset.from_tensor_slices(positive_images).take(300).shuffle(len(positive_images))
negative_dataset_path = tf.data.Dataset.from_tensor_slices(negative_images).take(300).shuffle(len(negative_images))

# %% [markdown]
# * visualize image in negative

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:11:46.367737Z","iopub.execute_input":"2024-08-02T06:11:46.368129Z","iopub.status.idle":"2024-08-02T06:11:46.376563Z","shell.execute_reply.started":"2024-08-02T06:11:46.368100Z","shell.execute_reply":"2024-08-02T06:11:46.375442Z"}}
print(len(negative_dataset_path))
print(negative_dataset_path.as_numpy_iterator().next())

# %% [markdown]
# ### Zip Dataset
# * (anchor, positive) => 1,1,1,1,1
# *  (anchor, negative) => 0,0,0,0,0

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:11:48.494054Z","iopub.execute_input":"2024-08-02T06:11:48.494454Z","iopub.status.idle":"2024-08-02T06:11:48.516846Z","shell.execute_reply.started":"2024-08-02T06:11:48.494423Z","shell.execute_reply":"2024-08-02T06:11:48.515643Z"}}
positives = tf.data.Dataset.zip(
    (anchor_dataset_path,
    positive_dataset_path,
tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor_dataset_path)))))

negatives = tf.data.Dataset.zip((anchor_dataset_path, negative_dataset_path, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor_dataset_path)))))

#dataset = positives.concatenate(negatives)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:11:50.691729Z","iopub.execute_input":"2024-08-02T06:11:50.692127Z","iopub.status.idle":"2024-08-02T06:11:50.698182Z","shell.execute_reply.started":"2024-08-02T06:11:50.692097Z","shell.execute_reply":"2024-08-02T06:11:50.697096Z"}}
#list(positives.as_numpy_iterator())
print(len(positives), len(negatives))

# %% [markdown]
# ## Visualize

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:12:06.526370Z","iopub.execute_input":"2024-08-02T06:12:06.526762Z","iopub.status.idle":"2024-08-02T06:12:06.533729Z","shell.execute_reply.started":"2024-08-02T06:12:06.526735Z","shell.execute_reply":"2024-08-02T06:12:06.532669Z"}}
def visualize(anchor, val, label):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))
    plt.suptitle(f"label : {label}", fontsize=16)
    axs = fig.subplots(1, 2)
    show(axs[0], anchor)
    show(axs[1], val)

# %% [markdown]
# ## Preprocessing Image

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T15:09:55.273407Z","iopub.execute_input":"2024-08-02T15:09:55.274140Z","iopub.status.idle":"2024-08-02T15:09:55.282733Z","shell.execute_reply.started":"2024-08-02T15:09:55.274103Z","shell.execute_reply":"2024-08-02T15:09:55.281174Z"}}
def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    #resize
    image = tf.image.resize(image, target_shape)
    image = tf.cast(image, tf.float32)
    
    return image


def preprocess_twin(anchor_img, val_img, label):
    return (
        preprocess_image(anchor_img),
        preprocess_image(val_img),
        label
    )

# %% [markdown]
# * test preprocess_image

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T03:18:18.611537Z","iopub.execute_input":"2024-08-02T03:18:18.611789Z","iopub.status.idle":"2024-08-02T03:18:18.616100Z","shell.execute_reply.started":"2024-08-02T03:18:18.611767Z","shell.execute_reply":"2024-08-02T03:18:18.615303Z"}}
# preprocess_image('/kaggle/input/dataset-piko-face/Dataset Piko Face/negative/Abdullah_al-Attiyah_0001.jpg')

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:12:25.773922Z","iopub.execute_input":"2024-08-02T06:12:25.774785Z","iopub.status.idle":"2024-08-02T06:12:25.877814Z","shell.execute_reply.started":"2024-08-02T06:12:25.774750Z","shell.execute_reply":"2024-08-02T06:12:25.876929Z"}}
test = positive_dataset_path.map(lambda x: preprocess_image(x), num_parallel_calls=tf.data.AUTOTUNE)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:12:26.699958Z","iopub.execute_input":"2024-08-02T06:12:26.700353Z","iopub.status.idle":"2024-08-02T06:12:26.755045Z","shell.execute_reply.started":"2024-08-02T06:12:26.700323Z","shell.execute_reply":"2024-08-02T06:12:26.753881Z"}}
test_example = test.as_numpy_iterator().next()

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:12:27.760334Z","iopub.execute_input":"2024-08-02T06:12:27.761086Z","iopub.status.idle":"2024-08-02T06:12:28.036330Z","shell.execute_reply.started":"2024-08-02T06:12:27.761046Z","shell.execute_reply":"2024-08-02T06:12:28.035183Z"}}
plt.imshow(test_example)

# %% [markdown]
# ## Mapping our dataset path to numpy array

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:12:34.608722Z","iopub.execute_input":"2024-08-02T06:12:34.609114Z","iopub.status.idle":"2024-08-02T06:12:34.694822Z","shell.execute_reply.started":"2024-08-02T06:12:34.609084Z","shell.execute_reply":"2024-08-02T06:12:34.693788Z"}}
positives = positives.map(preprocess_twin)
negatives = negatives.map(preprocess_twin)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:12:40.307409Z","iopub.execute_input":"2024-08-02T06:12:40.308366Z","iopub.status.idle":"2024-08-02T06:12:40.312332Z","shell.execute_reply.started":"2024-08-02T06:12:40.308332Z","shell.execute_reply":"2024-08-02T06:12:40.311202Z"}}
# list(ds.as_numpy_iterator())

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:12:40.464421Z","iopub.execute_input":"2024-08-02T06:12:40.465250Z","iopub.status.idle":"2024-08-02T06:12:40.718821Z","shell.execute_reply.started":"2024-08-02T06:12:40.465218Z","shell.execute_reply":"2024-08-02T06:12:40.715815Z"}}
example = positives.as_numpy_iterator().next()
visualize(*(example))

# %% [markdown]
# ## Augmentation

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:12:47.729524Z","iopub.execute_input":"2024-08-02T06:12:47.729926Z","iopub.status.idle":"2024-08-02T06:12:47.738142Z","shell.execute_reply.started":"2024-08-02T06:12:47.729897Z","shell.execute_reply":"2024-08-02T06:12:47.736991Z"}}
def data_aug(img):
    img = tf.image.stateless_random_brightness(img, max_delta=0.2, seed=(1,2))
    img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
    img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
    img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
    img = tf.image.stateless_random_saturation(img, lower=1,upper=1.5, seed=(np.random.randint(100),np.random.randint(100)))
    #img = keras.layers.RandomRotation(factor=0.2)(img)
    return img

def augment(anchor, val, label):
    aug_anchor = anchor
    aug_val = val
    return (data_aug(aug_anchor), data_aug(aug_val), label)

# %% [markdown]
# ### one image made 9 augment

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:12:50.272630Z","iopub.execute_input":"2024-08-02T06:12:50.273445Z","iopub.status.idle":"2024-08-02T06:12:51.694128Z","shell.execute_reply.started":"2024-08-02T06:12:50.273410Z","shell.execute_reply":"2024-08-02T06:12:51.692982Z"}}
positives_new = positives
for i in range(9):
    positive_aug = positives.map(augment, num_parallel_calls= tf.data.AUTOTUNE)
    positives_new = positives_new.concatenate(positive_aug)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:12:53.348825Z","iopub.execute_input":"2024-08-02T06:12:53.349837Z","iopub.status.idle":"2024-08-02T06:12:54.462090Z","shell.execute_reply.started":"2024-08-02T06:12:53.349797Z","shell.execute_reply":"2024-08-02T06:12:54.460930Z"}}
negatives_new = negatives
for i in range(9):
    negative_aug = negatives.map(augment, num_parallel_calls= tf.data.AUTOTUNE)
    negatives_new = negatives_new.concatenate(negative_aug)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:12:55.409017Z","iopub.execute_input":"2024-08-02T06:12:55.409405Z","iopub.status.idle":"2024-08-02T06:12:55.417998Z","shell.execute_reply.started":"2024-08-02T06:12:55.409376Z","shell.execute_reply":"2024-08-02T06:12:55.416818Z"}}
len(positives), len(positives_new), len(negatives), len(negatives_new)

# %% [markdown]
# ## Concat positive and negative together

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:13:13.587277Z","iopub.execute_input":"2024-08-02T06:13:13.588244Z","iopub.status.idle":"2024-08-02T06:13:13.597584Z","shell.execute_reply.started":"2024-08-02T06:13:13.588208Z","shell.execute_reply":"2024-08-02T06:13:13.596533Z"}}
dataset = positives_new.concatenate(negatives_new)
len(dataset)

# %% [markdown]
# ## Preprocessing Pipeline

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:13:16.043979Z","iopub.execute_input":"2024-08-02T06:13:16.044466Z","iopub.status.idle":"2024-08-02T06:13:16.058032Z","shell.execute_reply.started":"2024-08-02T06:13:16.044431Z","shell.execute_reply":"2024-08-02T06:13:16.056784Z"}}
dataset = dataset.cache()
dataset = dataset.shuffle(buffer_size=10000)

# %% [markdown]
# ## Split dataset to train and val

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:13:18.754894Z","iopub.execute_input":"2024-08-02T06:13:18.755993Z","iopub.status.idle":"2024-08-02T06:13:18.770239Z","shell.execute_reply.started":"2024-08-02T06:13:18.755954Z","shell.execute_reply":"2024-08-02T06:13:18.769154Z"}}
image_count = len(dataset)
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))
test_dataset = val_dataset.take(round(len(val_dataset) * 0.5))

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:13:28.608373Z","iopub.execute_input":"2024-08-02T06:13:28.608758Z","iopub.status.idle":"2024-08-02T06:13:28.616828Z","shell.execute_reply.started":"2024-08-02T06:13:28.608731Z","shell.execute_reply":"2024-08-02T06:13:28.615720Z"}}
len(train_dataset), len(val_dataset), len(test_dataset)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:13:37.412364Z","iopub.execute_input":"2024-08-02T06:13:37.412763Z","iopub.status.idle":"2024-08-02T06:13:37.430486Z","shell.execute_reply.started":"2024-08-02T06:13:37.412734Z","shell.execute_reply":"2024-08-02T06:13:37.429161Z"}}
# Training partition
train_dataset = train_dataset.batch(16)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

# Val partition
val_dataset = val_dataset.batch(16)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

#test partition
test_dataset = test_dataset.batch(16)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# %% [markdown]
# ## Model Engineering

# %% [markdown]
# ### Build Embedding Layer

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:13:51.221174Z","iopub.execute_input":"2024-08-02T06:13:51.222461Z","iopub.status.idle":"2024-08-02T06:13:51.230833Z","shell.execute_reply.started":"2024-08-02T06:13:51.222416Z","shell.execute_reply":"2024-08-02T06:13:51.229341Z"}}
def make_embedding(): 
    inp = Input(shape=target_shape + (3,), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:13:58.251347Z","iopub.execute_input":"2024-08-02T06:13:58.251743Z","iopub.status.idle":"2024-08-02T06:13:58.637073Z","shell.execute_reply.started":"2024-08-02T06:13:58.251715Z","shell.execute_reply":"2024-08-02T06:13:58.636003Z"}}
embedding = make_embedding()
embedding.summary()

# %% [markdown]
# ### Build Distance Layer

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T15:10:13.615531Z","iopub.execute_input":"2024-08-02T15:10:13.615965Z","iopub.status.idle":"2024-08-02T15:10:13.624742Z","shell.execute_reply.started":"2024-08-02T15:10:13.615934Z","shell.execute_reply":"2024-08-02T15:10:13.623359Z"}}
# Siamese L1 Distance class
class DistanceLayer(Layer):
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        
        return ops.absolute((input_embedding[0] - validation_embedding[0]))

# %% [markdown]
# ### Make Siamese Model

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:14:19.198908Z","iopub.execute_input":"2024-08-02T06:14:19.199986Z","iopub.status.idle":"2024-08-02T06:14:19.207575Z","shell.execute_reply.started":"2024-08-02T06:14:19.199943Z","shell.execute_reply":"2024-08-02T06:14:19.206345Z"}}
def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=target_shape + (3,) )
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=target_shape + (3,) )
    
    # Combine siamese distance components
    siamese_layer = DistanceLayer()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:14:27.690515Z","iopub.execute_input":"2024-08-02T06:14:27.691544Z","iopub.status.idle":"2024-08-02T06:14:27.736306Z","shell.execute_reply.started":"2024-08-02T06:14:27.691508Z","shell.execute_reply":"2024-08-02T06:14:27.735318Z"}}
siamese_model = make_siamese_model()
siamese_model.summary()

# %% [markdown]
# ## Training

# %% [markdown]
# ### Setup Loss and Optimizer

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:14:50.557220Z","iopub.execute_input":"2024-08-02T06:14:50.557732Z","iopub.status.idle":"2024-08-02T06:14:50.606044Z","shell.execute_reply.started":"2024-08-02T06:14:50.557696Z","shell.execute_reply":"2024-08-02T06:14:50.605101Z"}}
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = keras.optimizers.Adam(1e-4) # 0.0001

# %% [markdown]
# ### Establish Checkpoints

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:15:37.720197Z","iopub.execute_input":"2024-08-02T06:15:37.720613Z","iopub.status.idle":"2024-08-02T06:15:37.725824Z","shell.execute_reply.started":"2024-08-02T06:15:37.720582Z","shell.execute_reply":"2024-08-02T06:15:37.724590Z"}}
os.mkdir('/kaggle/working/checkpoint')

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:15:44.913328Z","iopub.execute_input":"2024-08-02T06:15:44.914149Z","iopub.status.idle":"2024-08-02T06:15:44.919597Z","shell.execute_reply.started":"2024-08-02T06:15:44.914115Z","shell.execute_reply":"2024-08-02T06:15:44.918466Z"}}
checkpoint_dir = '/kaggle/working/checkpoint/'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# %% [markdown]
# ### Build Train Step Function

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:16:01.174988Z","iopub.execute_input":"2024-08-02T06:16:01.175408Z","iopub.status.idle":"2024-08-02T06:16:13.340856Z","shell.execute_reply.started":"2024-08-02T06:16:01.175379Z","shell.execute_reply":"2024-08-02T06:16:13.339731Z"}}
test_batch = train_dataset.as_numpy_iterator()
batch_1 = test_batch.next()
X = batch_1[:2]
y = batch_1[2]

len(train_dataset), y

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:16:53.350362Z","iopub.execute_input":"2024-08-02T06:16:53.351072Z","iopub.status.idle":"2024-08-02T06:16:53.500403Z","shell.execute_reply.started":"2024-08-02T06:16:53.351040Z","shell.execute_reply":"2024-08-02T06:16:53.499275Z"}}
visualize(batch_1[0][2], batch_1[1][2], batch_1[2][2]) 

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:16:57.652039Z","iopub.execute_input":"2024-08-02T06:16:57.652452Z","iopub.status.idle":"2024-08-02T06:16:57.659900Z","shell.execute_reply.started":"2024-08-02T06:16:57.652422Z","shell.execute_reply":"2024-08-02T06:16:57.658735Z"}}
@tf.function
def train_step(batch):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    # Return loss
    return loss

# %% [markdown]
# ### Build Training Loop

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:17:07.940796Z","iopub.execute_input":"2024-08-02T06:17:07.941707Z","iopub.status.idle":"2024-08-02T06:17:07.947885Z","shell.execute_reply.started":"2024-08-02T06:17:07.941669Z","shell.execute_reply":"2024-08-02T06:17:07.946851Z"}}
# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:17:29.028613Z","iopub.execute_input":"2024-08-02T06:17:29.029022Z","iopub.status.idle":"2024-08-02T06:17:29.037869Z","shell.execute_reply.started":"2024-08-02T06:17:29.028993Z","shell.execute_reply":"2024-08-02T06:17:29.036532Z"}}
def train(data, EPOCHS):
    # Loop through epochs
    with tf.device('/gpu:0'):
        for epoch in range(1, EPOCHS+1):
            print('\n Epoch {}/{}'.format(epoch, EPOCHS))
            progbar = tf.keras.utils.Progbar(len(data))

            # Creating a metric object 
            r = Recall()
            p = Precision()

            # Loop through each batch
            for idx, batch in enumerate(data):
                # Run train step here
                loss = train_step(batch)
                yhat = siamese_model.predict(batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat) 
                progbar.update(idx+1)
            print(loss.numpy(), r.result().numpy(), p.result().numpy())

            # Save checkpoints
            if epoch % 10 == 0: 
                checkpoint.save(file_prefix=checkpoint_prefix)

# %% [markdown]
# ### Train the model

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:17:34.013289Z","iopub.execute_input":"2024-08-02T06:17:34.014017Z","iopub.status.idle":"2024-08-02T06:17:34.020706Z","shell.execute_reply.started":"2024-08-02T06:17:34.013985Z","shell.execute_reply":"2024-08-02T06:17:34.019573Z"}}
len(train_dataset)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T06:17:42.279216Z","iopub.execute_input":"2024-08-02T06:17:42.280215Z","iopub.status.idle":"2024-08-02T06:18:06.373735Z","shell.execute_reply.started":"2024-08-02T06:17:42.280183Z","shell.execute_reply":"2024-08-02T06:18:06.372070Z"}}
train(train_dataset, epochs)

# %% [markdown]
# ## Evaluate Model

# %% [markdown]
# ### Make Predictions

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T05:37:23.479632Z","iopub.execute_input":"2024-08-02T05:37:23.480010Z","iopub.status.idle":"2024-08-02T05:37:23.947032Z","shell.execute_reply.started":"2024-08-02T05:37:23.479981Z","shell.execute_reply":"2024-08-02T05:37:23.946195Z"}}
# Get a batch of test data
test_input, test_val, y_true = test_dataset.as_numpy_iterator().next()

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T05:37:27.453953Z","iopub.execute_input":"2024-08-02T05:37:27.454325Z","iopub.status.idle":"2024-08-02T05:37:28.074722Z","shell.execute_reply.started":"2024-08-02T05:37:27.454295Z","shell.execute_reply":"2024-08-02T05:37:28.073804Z"}}
y_hat = siamese_model.predict([test_input, test_val])

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T05:37:30.733770Z","iopub.execute_input":"2024-08-02T05:37:30.734404Z","iopub.status.idle":"2024-08-02T05:37:30.741342Z","shell.execute_reply.started":"2024-08-02T05:37:30.734367Z","shell.execute_reply":"2024-08-02T05:37:30.740300Z"}}
# Post processing the results 
[1 if prediction > 0.5 else 0 for prediction in y_hat ]

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T05:37:34.856611Z","iopub.execute_input":"2024-08-02T05:37:34.857337Z","iopub.status.idle":"2024-08-02T05:37:34.863550Z","shell.execute_reply.started":"2024-08-02T05:37:34.857305Z","shell.execute_reply":"2024-08-02T05:37:34.862505Z"}}
y_true

# %% [markdown]
# ### Calculate Metrics

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T05:37:50.197815Z","iopub.execute_input":"2024-08-02T05:37:50.198632Z","iopub.status.idle":"2024-08-02T05:37:50.234328Z","shell.execute_reply.started":"2024-08-02T05:37:50.198602Z","shell.execute_reply":"2024-08-02T05:37:50.233403Z"}}
# Creating a metric object 
m_recall = Recall()

# Calculating the recall value 
m_recall.update_state(y_true, y_hat)

# Creating a metric object 
m_precision = Precision()

# Calculating the recall value 
m_precision.update_state(y_true, y_hat)

# Return Recall Result
print(f'Recall : {m_recall.result().numpy()}')

# Return Recall Result
print(f'Precision : {m_precision.result().numpy()}')

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T05:37:56.215916Z","iopub.execute_input":"2024-08-02T05:37:56.216617Z","iopub.status.idle":"2024-08-02T05:38:22.028335Z","shell.execute_reply.started":"2024-08-02T05:37:56.216588Z","shell.execute_reply":"2024-08-02T05:38:22.027369Z"},"jupyter":{"outputs_hidden":true}}
r = Recall()
p = Precision()

for test_input, test_val, y_true in test_dataset.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat) 

print(r.result().numpy(), p.result().numpy())

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T05:41:50.094172Z","iopub.execute_input":"2024-08-02T05:41:50.094796Z","iopub.status.idle":"2024-08-02T05:41:50.101557Z","shell.execute_reply.started":"2024-08-02T05:41:50.094766Z","shell.execute_reply":"2024-08-02T05:41:50.100550Z"}}
[1 if prediction > 0.5 else 0 for prediction in y_hat ]

# %% [markdown]
# ## Viz Results

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T05:43:37.924977Z","iopub.execute_input":"2024-08-02T05:43:37.925364Z","iopub.status.idle":"2024-08-02T05:43:38.432561Z","shell.execute_reply.started":"2024-08-02T05:43:37.925325Z","shell.execute_reply":"2024-08-02T05:43:38.431564Z"}}
# Set plot size 
plt.figure(figsize=(10,8))

# Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[2])

# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[2])

# Renders cleanly
plt.show()

# %% [markdown]
# ## Save Model

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T05:38:54.498851Z","iopub.execute_input":"2024-08-02T05:38:54.499262Z","iopub.status.idle":"2024-08-02T05:38:57.700117Z","shell.execute_reply.started":"2024-08-02T05:38:54.499229Z","shell.execute_reply":"2024-08-02T05:38:57.699268Z"}}
# Save weights
siamese_model.save('siamesemodelv2.h5')

# %% [markdown]
# ## Load Model

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T15:11:07.595727Z","iopub.execute_input":"2024-08-02T15:11:07.596134Z","iopub.status.idle":"2024-08-02T15:11:10.083138Z","shell.execute_reply.started":"2024-08-02T15:11:07.596103Z","shell.execute_reply":"2024-08-02T15:11:10.081812Z"}}
# Reload model 
siamese_model = keras.models.load_model('/kaggle/input/face/tensorflow2/default/1/siamesemodelv2.h5', 
                                   custom_objects={'DistanceLayer':DistanceLayer, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T08:04:46.602944Z","iopub.execute_input":"2024-08-02T08:04:46.604298Z","iopub.status.idle":"2024-08-02T08:04:46.623409Z","shell.execute_reply.started":"2024-08-02T08:04:46.604215Z","shell.execute_reply":"2024-08-02T08:04:46.622079Z"}}
siamese_model

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T15:11:13.824190Z","iopub.execute_input":"2024-08-02T15:11:13.824569Z","iopub.status.idle":"2024-08-02T15:11:13.851583Z","shell.execute_reply.started":"2024-08-02T15:11:13.824541Z","shell.execute_reply":"2024-08-02T15:11:13.849888Z"}}
siamese_model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T15:10:41.707697Z","iopub.execute_input":"2024-08-02T15:10:41.708268Z","iopub.status.idle":"2024-08-02T15:10:42.033766Z","shell.execute_reply.started":"2024-08-02T15:10:41.708228Z","shell.execute_reply":"2024-08-02T15:10:42.032364Z"}}
test = preprocess_image('/kaggle/input/testing/input_img.jpg')

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T15:10:49.370610Z","iopub.execute_input":"2024-08-02T15:10:49.371001Z","iopub.status.idle":"2024-08-02T15:10:49.660512Z","shell.execute_reply.started":"2024-08-02T15:10:49.370970Z","shell.execute_reply":"2024-08-02T15:10:49.658781Z"}}
plt.imshow(test)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T15:15:36.741372Z","iopub.execute_input":"2024-08-02T15:15:36.742451Z","iopub.status.idle":"2024-08-02T15:15:37.504770Z","shell.execute_reply.started":"2024-08-02T15:15:36.742405Z","shell.execute_reply":"2024-08-02T15:15:37.503535Z"}}
positive_test = preprocess_image('/kaggle/input/dataset-piko-face/Dataset Piko Face/positive/16cc5fdc-4f14-11ef-9794-beecda8d9db4.jpg')
plt.imshow(positive_test)
print(positive_test.shape)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-02T15:15:04.275543Z","iopub.execute_input":"2024-08-02T15:15:04.275938Z","iopub.status.idle":"2024-08-02T15:15:04.507916Z","shell.execute_reply.started":"2024-08-02T15:15:04.275911Z","shell.execute_reply":"2024-08-02T15:15:04.506416Z"},"jupyter":{"outputs_hidden":true}}
y_hat = siamese_model.predict([test, positive_test])

# %% [code]
