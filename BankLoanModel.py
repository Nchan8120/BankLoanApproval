#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# In[37]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[38]:


# Load the data
df = pd.read_csv('bankloan.csv')

# Basic Info
df.head()


# In[39]:


df.describe()


# In[40]:


df.isnull().sum()


# In[41]:


# Unique values in categorical columns
categorical_cols = ['ZIP.Code', 'Family', 'Education', 'Personal.Loan',
                    'Securities.Account', 'CD.Account', 'Online', 'CreditCard']


# In[42]:


# Count plots for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, palette='Set2')
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[43]:


# Distribution plots for continuous variables
continuous_cols = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage']
for col in continuous_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()


# # Cleaning the Data

# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# In[ ]:





# In[45]:


# Drop ID and ZipCode (not useful for prediction)
df.drop(['ID', 'ZIP.Code'], axis=1, inplace=True)


# In[46]:


# Features and Target
X = df.drop('Personal.Loan', axis=1)
y = df['Personal.Loan']


# In[47]:


# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[48]:


# Save the fitted scaler - to use for streamlit UI
import joblib
joblib.dump(scaler, "scaler.joblib")


# In[49]:


# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# In[50]:


print("Data is ready.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train distribution:\n", y_train.value_counts(normalize=True))


# # Building Model with Keras

# In[51]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.utils.class_weight import compute_class_weight


# In[52]:


# Class weight works only with numpy arrays
y_train = y_train.to_numpy()


# In[53]:


# Compute class weights to handle imbalance
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

# Properly map weights to actual class labels
class_weight_dict = dict(zip(classes, class_weights))

print("Class weights:", class_weight_dict)


# In[54]:


# Build the model with dropout to prevent overfitting
model = Sequential([
    Input(shape=(X_train.shape[1],)), 
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])


# In[55]:


# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])


# In[56]:


# Define callbacks for early stopping, model checkpointing and logging

# Callbacks
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("bankloan_model.keras", save_best_only=True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="./logs/loan_model")

callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]


# In[57]:


# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)


# # Evaluating the Model

# In[58]:


loss, accuracy, auc = model.evaluate(X_test, y_test.to_numpy(), verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")


# In[59]:


y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)


# In[60]:


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[61]:


plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')           # X-axis label
plt.ylabel('Accuracy')        # Y-axis label
plt.title('Model Accuracy')   # Plot title
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')           # X-axis label
plt.ylabel('Loss')            # Y-axis label
plt.title('Model Loss')       # Plot title
plt.legend()
plt.show()


# # Saving Model

# In[62]:


model.save("bankloan_model.keras") 


# # Interpreting Results

# ### Test Metrics
# - Test Loss: 0.1086 — Low, which indicates good performance.
# - Test Accuracy: 95.7% — Excellent overall accuracy.
# - Test AUC: 0.9940 — Outstanding. This indicates your model separates the classes very well.
# 
# ### Training/Validation Accuracy Plot
# - The training and validation accuracy curves are close together and steadily increase.
# - There's no major gap or divergence → No overfitting is apparent.
# - Training accuracy approaches ≈ 95%+, which matches validation accuracy meaning this is healthy training
# 
# ### Training/Validation Loss Plot
# - Loss decreases consistently for both training and validation.
# - Validation loss fluctuates slightly but continues to drop → Suggests good generalization.
# - No sharp rise in validation loss means no overfitting.
# 
# ### Classification Report
# - Class 0: Perfect precision — no false positives, and high recall.
# - Class 1: Good recall (97%) — you’re catching almost all true positives.
# - Precision is lower (70%) → means you're getting some false positives on class 1.
# - F1-score of 0.81 for class 1 is solid for an imbalanced dataset.
# 
# ### Confusion Matrix
# - 3 false negatives (missed positives)
# - 40 false positives (predicted positive when it's negative) — this is why the precision dips

# # Report

# # Lessons to Learn
# 
# Nathan: Not every technique is universally applicable. LSTMs, while powerful for sequential data like time series or text, are not well-suited for tabular datasets like structured loan applications. Batch Normalization, often used in deep convolutional networks, did not significantly improve performance in this simpler fully connected architecture, and in some cases, even slowed training. In future projects, I’ll be more intentional about selecting techniques based on the nature of the problem rather than the novelty of the method.

# 

# In[ ]:




