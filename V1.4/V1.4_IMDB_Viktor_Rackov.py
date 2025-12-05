#!/usr/bin/env python
# coding: utf-8

# # V1.4
# **Avtor:** Viktor Rackov

# In[9]:


import os, numpy as np, matplotlib.pyplot as plt
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

NUM_WORDS = 3000
VAL_SPLIT = 10000
EPOCHS = 30
BATCH_SIZE = 512
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
print(tf.__version__)


# ## Nalaganje in vektorizacija (multi-hot, top 3000)
# 

# In[10]:


def vectorize_sequences(sequences, dimension=NUM_WORDS):
    result = np.zeros((len(sequences), dimension), dtype="float32")
    for i, seq in enumerate(sequences):
        idx = np.unique([w for w in seq if 0 <= w < dimension])
        result[i, idx] = 1.0
    return result

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
x_train_vec = vectorize_sequences(x_train, NUM_WORDS)
x_test_vec  = vectorize_sequences(x_test, NUM_WORDS)
x_train_vec.shape, x_test_vec.shape


# ## Model s 4 nevroni v skritih plasteh
# 

# In[ ]:


x_val, y_val = x_train_vec[:VAL_SPLIT], y_train[:VAL_SPLIT]
partial_x_train, partial_y_train = x_train_vec[VAL_SPLIT:], y_train[VAL_SPLIT:]

model = keras.Sequential([
    keras.Input(shape=(NUM_WORDS,)),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# ## Učenje
# 

# In[ ]:


early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    partial_x_train, partial_y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    verbose=2
)


# ## Grafi izgube in točnosti
# 

# In[ ]:


hist = history.history
# Loss
plt.figure()
plt.plot(hist['loss'], label='train_loss')
plt.plot(hist['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss (BCE)'); plt.title('Training vs Validation Loss (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# Accuracy
plt.figure()
plt.plot(hist['accuracy'], label='train_acc')
plt.plot(hist['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()


# ## Detekcija prekomernega prileganja in testna ocena
# 

# In[ ]:


val_loss = np.array(hist['val_loss'], dtype=float)
loss = np.array(hist['loss'], dtype=float)

best_epoch = int(np.argmin(val_loss)) + 1
start_epoch = None
for i in range(1, len(val_loss)):
    if (val_loss[i] > val_loss[i-1]) and (loss[i] < loss[i-1]):
        start_epoch = i + 1
        break

print(f"Najboljši epoch po val_loss: {best_epoch}")
if start_epoch is not None:
    print(f"Prekomerno prileganje se začne približno pri epohi: {start_epoch}")
else:
    print("Jasnega začetka prekomernega prileganja v znotraj treniranih epoh ni.")

test_loss, test_acc = model.evaluate(x_test_vec, y_test, verbose=0)
print(f"Test — loss: {test_loss:.4f}, acc: {test_acc:.4f}")


# # 512 WITH DROPOUT
# 

# In[19]:


model = keras.Sequential([
    keras.Input(shape=(NUM_WORDS,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[20]:


early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    partial_x_train, partial_y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    verbose=2
)


# In[21]:


hist = history.history
# Loss
plt.figure()
plt.plot(hist['loss'], label='train_loss')
plt.plot(hist['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss (BCE)'); plt.title('Training vs Validation Loss (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# Accuracy
plt.figure()
plt.plot(hist['accuracy'], label='train_acc')
plt.plot(hist['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()


# # 512 both

# In[22]:


model = keras.Sequential([
    keras.Input(shape=(NUM_WORDS,)),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.6),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.6),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[23]:


early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    partial_x_train, partial_y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    verbose=2
)


# In[24]:


hist = history.history
# Loss
plt.figure()
plt.plot(hist['loss'], label='train_loss')
plt.plot(hist['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss (BCE)'); plt.title('Training vs Validation Loss (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# Accuracy
plt.figure()
plt.plot(hist['accuracy'], label='train_acc')
plt.plot(hist['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()


# # 16 NN

# In[26]:


model = keras.Sequential([
    keras.Input(shape=(NUM_WORDS,)),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    # layers.Dropout(0.6),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    # layers.Dropout(0.6),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    partial_x_train, partial_y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    verbose=2
)

hist = history.history
# Loss
plt.figure()
plt.plot(hist['loss'], label='train_loss')
plt.plot(hist['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss (BCE)'); plt.title('Training vs Validation Loss (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# Accuracy
plt.figure()
plt.plot(hist['accuracy'], label='train_acc')
plt.plot(hist['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()



# In[27]:


model = keras.Sequential([
    keras.Input(shape=(NUM_WORDS,)),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    partial_x_train, partial_y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    verbose=2
)

hist = history.history
# Loss
plt.figure()
plt.plot(hist['loss'], label='train_loss')
plt.plot(hist['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss (BCE)'); plt.title('Training vs Validation Loss (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# Accuracy
plt.figure()
plt.plot(hist['accuracy'], label='train_acc')
plt.plot(hist['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()



# # 16 NN with both L2 Regularization and dropout
# 

# In[28]:


model = keras.Sequential([
    keras.Input(shape=(NUM_WORDS,)),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.6),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.6),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    partial_x_train, partial_y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    verbose=2
)

hist = history.history
# Loss
plt.figure()
plt.plot(hist['loss'], label='train_loss')
plt.plot(hist['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss (BCE)'); plt.title('Training vs Validation Loss (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# Accuracy
plt.figure()
plt.plot(hist['accuracy'], label='train_acc')
plt.plot(hist['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()



# # NN 4 w/ regulizers

# In[29]:


model = keras.Sequential([
    keras.Input(shape=(NUM_WORDS,)),
    layers.Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    # layers.Dropout(0.6),
    layers.Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    # layers.Dropout(0.6),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    partial_x_train, partial_y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    verbose=2
)

hist = history.history
# Loss
plt.figure()
plt.plot(hist['loss'], label='train_loss')
plt.plot(hist['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss (BCE)'); plt.title('Training vs Validation Loss (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# Accuracy
plt.figure()
plt.plot(hist['accuracy'], label='train_acc')
plt.plot(hist['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()



# # NN4 dropout

# In[31]:


model = keras.Sequential([
    keras.Input(shape=(NUM_WORDS,)),
    layers.Dense(4, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    partial_x_train, partial_y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    verbose=2
)

hist = history.history
# Loss
plt.figure()
plt.plot(hist['loss'], label='train_loss')
plt.plot(hist['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss (BCE)'); plt.title('Training vs Validation Loss (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# Accuracy
plt.figure()
plt.plot(hist['accuracy'], label='train_acc')
plt.plot(hist['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()



# # NN 4 BOTH

# In[32]:


model = keras.Sequential([
    keras.Input(shape=(NUM_WORDS,)),
    layers.Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    partial_x_train, partial_y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    verbose=2
)

hist = history.history
# Loss
plt.figure()
plt.plot(hist['loss'], label='train_loss')
plt.plot(hist['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss (BCE)'); plt.title('Training vs Validation Loss (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()

# Accuracy
plt.figure()
plt.plot(hist['accuracy'], label='train_acc')
plt.plot(hist['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy (4 units)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()



# ### 512 neurons
# 
# L2 (λ=0.001): Training becomes more volatile; overfitting is still present, just delayed a bit.
# 
# Dropout (p=0.6): Behavior is similar to the baseline, but with slower, smoother changes in the curves.
# 
# L2 + Dropout: Still shows sharp jumps and slow trend changes; overfitting persists → the network is too large and doesn’t generalize well.
# 
# ### 16 neurons
# 
# L2 (λ=0.001): Slightly better than the baseline; overfitting rises more slowly, but is still substantial.
# 
# Dropout (p≈0.5–0.6): Similar benefits as with 512: slower dynamics, later onset of overfitting; looks more stable but still overfits.
# 
# L2 + Dropout: Best among the 16-unit variants—most stable and least overfitting, though not entirely removed.
# 
# ### 4 neurons
# 
# L2 (λ=0.001): Slightly worse than plain 4-unit.
# 
# Dropout: Better than plain 4-unit—similar qualitative effect as with larger nets (smoother, slower), but capacity is still low.
# 
# L2 + Dropout: Learns slowly and steadily; loss decreases smoothly, no spikes, no clear overfitting
# 
# 
# 
# Validation accuracy changes are minimal across all models
# 

# 
