import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense



# Define the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=seq_length),
    LSTM(units=units, return_sequences=True),
    LSTM(units=units),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# Generate text
def generate_text(model, seed_text, next_words):
  for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=seq_length-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted, axis=1)
    output_word = tokenizer.index_word[predicted_index[0]]
    seed_text += " " + output_word
  return seed_text

# Example usage
seed_text = "The quick brown fox"
next_words = 10
generated_text = generate_text(model, seed_text, next_words)
print(generated_text)