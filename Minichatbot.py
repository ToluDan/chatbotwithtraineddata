# Importing required libraries
import json, os, nltk, random, torch
import numpy as np
import torch.nn as nn

# DataLoader helps train data in batches
from torch.utils.data import DataLoader, TensorDataset

# Optimizer for training
import torch.optim as optim

# Functional utilities like softmax
import torch.nn.functional as F

# Used to reduce words to their base form
from nltk.stem import WordNetLemmatizer


# Download required nltk packages
# punkt -> tokenizer
# wordnet -> dictionary for lemmatization
# omw-1.4 -> multilingual support for wordnet
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# =========================
# NEURAL NETWORK MODEL
# =========================
class ChatbotModel(nn.Module):

    """
    This class defines the neural network architecture
    used to classify user messages into intents.
    """

    def __init__(self, input_size, output_size):

        """
        input_size  -> number of vocabulary words
        output_size -> number of intents/classes
        """

        super(ChatbotModel, self).__init__()

        # First hidden layer
        self.fc1 = nn.Linear(input_size, 128)

        # Second hidden layer
        self.fc2 = nn.Linear(128, 64)

        # Final output layer
        self.fc3 = nn.Linear(64, output_size)

        # Activation function
        self.relu = nn.ReLU()

        # Dropout helps reduce overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        """
        Defines how data flows through the network.
        """

        # Pass input through first layer + activation
        x = self.relu(self.fc1(x))

        # Randomly deactivate neurons during training
        x = self.dropout(x)

        # Pass through second layer
        x = self.relu(self.fc2(x))

        # Apply dropout again
        x = self.dropout(x)

        # Final output scores
        x = self.fc3(x)

        return x


# =========================
# CHATBOT ASSISTANT CLASS
# =========================
class ChatBotAssistant():

    """
    Main chatbot class responsible for:
    - loading intents
    - preprocessing text
    - preparing training data
    - training model
    - saving/loading model
    - generating responses
    """

    def __init__(self, intents_paths, function_mapping=None):

        """
        intents_paths    -> path to intents JSON file
        function_mapping -> maps intents to Python functions
        """

        # Neural network model
        self.model = None

        # JSON file containing intents
        self.intents_path = intents_paths

        # Stores tokenized patterns + tags
        self.documents = []

        # Stores all unique words
        self.vocabulary = []

        # Stores intent names
        self.intents = []

        # Stores responses for each intent
        self.intents_responses = {}

        # Optional mapping of intents to functions
        self.function_mapping = function_mapping

        # Training features
        self.X = None

        # Training labels
        self.y = None

    # =========================
    # TOKENIZATION + LEMMATIZATION
    # =========================
    @staticmethod
    def token_and_lam(text):

        """
        Converts sentence into tokens
        and reduces words to root/base form.

        Example:
        giving -> give
        """

        lemmatizer = nltk.WordNetLemmatizer()

        # Split sentence into words
        words = nltk.word_tokenize(text)

        # Convert to lowercase and lemmatize
        words = [lemmatizer.lemmatize(word.lower(), pos='v') 
                 for word in words
                 if word.isalnum()
                 ]

        return words


    # =========================
    # BAG OF WORDS CREATION
    # =========================
    def word_bags(self, words):

        """
        Converts tokenized words into bag-of-words vector.

        Example:
        vocabulary = ['hello','bye','thanks']

        input = ['hello']

        output = [1,0,0]
        """

        return [1 if word in words else 0 for word in self.vocabulary]


    # =========================
    # LOAD AND PROCESS INTENTS
    # =========================
    def parse_intents(self):

        """
        Reads intents JSON file and:
        - extracts patterns
        - tokenizes patterns
        - builds vocabulary
        - stores responses
        """

        if os.path.exists(self.intents_path):

            with open(self.intents_path, 'r') as f:

                intents_data = json.load(f)

                # Loop through each intent
                for intent in intents_data['intents']:

                    # Avoid duplicate intents
                    if intent['tag'] not in self.intents:

                        # Save intent tag
                        self.intents.append(intent['tag'])

                        # Save responses
                        self.intents_responses[intent['tag']] = intent['responses']

                        # Process all patterns
                        for pattern in intent['patterns']:

                            # Tokenize and lemmatize
                            pattern_words = self.token_and_lam(pattern)

                            # Add words to vocabulary
                            self.vocabulary.extend(pattern_words)

                            # Save tokenized pattern + tag
                            self.documents.append((pattern_words, intent['tag']))

                        # Remove duplicate words and sort alphabetically
                        self.vocabulary = sorted(set(self.vocabulary))


    # =========================
    # PREPARE TRAINING DATA
    # =========================
    def prepare_data(self):

        """
        Converts text data into numerical format
        suitable for neural network training.
        """

        bags = []
        indices = []

        # Loop through tokenized documents
        for document in self.documents:

            words = document[0]

            # Convert words into bag-of-words vector
            bag = self.word_bags(words)

            # Convert intent tag into index number
            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        # Convert to numpy arrays
        self.X = np.array(bags)
        self.y = np.array(indices)


    # =========================
    # TRAIN MODEL
    # =========================
    def train_data(self, batch_size, lr, epochs):

        """
        Trains the neural network.
        """

        # Convert numpy arrays into tensors
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        # Create dataset object
        dataset = TensorDataset(X_tensor, y_tensor)

        # Create batches and shuffle data
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):

            running_loss = 0.0

            for batch_X, batch_y in loader:

                # Clear previous gradients
                optimizer.zero_grad()

                # Predict outputs
                output = self.model(batch_X)

                # Calculate error
                loss = criterion(output, batch_y)

                # Backpropagation
                loss.backward()

                # Update weights
                optimizer.step()

                running_loss += loss

            # Print epoch loss
            print(f"Epoch {epoch+1}: Loss {running_loss/len(loader):.4f}")


    # =========================
    # SAVE TRAINED MODEL
    # =========================
    def save_my_model(self, model_path, dimension_path):

        """
        Saves:
        - trained weights
        - input/output dimensions
        """

        torch.save(self.model.state_dict(), model_path)

        with open(dimension_path, 'w') as f:

            json.dump(
                {
                    'input_size': self.X.shape[1],
                    'output_size': len(self.intents)
                },
                f
            )


    # =========================
    # LOAD TRAINED MODEL
    # =========================
    def load_model(self, model_path, dimensions_path):

        """
        Loads saved model weights.
        """

        with open(dimensions_path, 'r') as f:

            dimensions = json.load(f)

            # Rebuild model architecture
            self.model = ChatbotModel(
                dimensions['input_size'],
                dimensions['output_size']
            )

            # Load saved weights
            self.model.load_state_dict(
                torch.load(model_path, weights_only=True)
            )


    # =========================
    # PROCESS USER MESSAGE
    # =========================
    def to_process_message(self, input_messages):

        """
        Predicts the intent of user message
        and returns a response.
        """

        # Tokenize message
        words = self.token_and_lam(input_messages)

        # Convert into bag-of-words vector
        # BUG FIX:
        # You mistakenly passed input_messages instead of words
        bags = self.word_bags(words)

        # Convert into tensor
        bag_tensor = torch.tensor([bags], dtype=torch.float32)

        # Disable gradient calculation
        self.model.eval()

        with torch.no_grad():

            # Predict
            predictions = self.model(bag_tensor)

            # Convert logits to probabilities
            probabilities = F.softmax(predictions, dim=1)

            # Get confidence score
            confidence = torch.max(probabilities).item()

        # Get predicted intent index
        predicted_class_index = torch.argmax(predictions, dim=1).item()

        # Convert index back to intent name
        predicted_intent = self.intents[predicted_class_index]

        # Confidence threshold
        if confidence < 0.70:
            return "I don't understand that yet."

        # Execute mapped function if available
        if self.function_mapping:

            if predicted_intent in self.function_mapping:

                return self.function_mapping[predicted_intent]()

        # Return random response from intent
        if self.intents_responses[predicted_intent]:

            return random.choice(
                self.intents_responses[predicted_intent]
            )


# =========================
# EXTERNAL FUNCTION
# =========================
def get_stocks():

    """
    Returns 3 random stock tickers.
    """

    stocks = ['META', 'AAPL', 'MSFT', 'NVDA', 'GS']

    return random.sample(stocks, 3)


# =========================
# MAIN PROGRAM
# =========================
if __name__ == '__main__':

    """
    Entry point of the chatbot application.
    """

    assistant = ChatBotAssistant('responseplusquestions.json', function_mapping= {'stocks':get_stocks})
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_data (batch_size =8, lr=0.001, epochs =100) 

    assistant.save_my_model('chatbot_model.pth','dimesions.json')

    # Create chatbot object
    # assistant = ChatBotAssistant(
    #     'responseplusquestions.json',
    #     function_mapping={'stocks': get_stocks}
    # )

    # # Load intents and vocabulary
    # assistant.parse_intents()

    # # Load saved trained model
    # assistant.load_model(
    #     'chatbot_model.pth',
    #     'dimesions.json'
    # )

    # Chat loop
    while True:

        # Get user input
        message = input("You: ")

        # Exit condition
        if message.lower() == "quit":
            break

        # Generate chatbot response
        response = assistant.to_process_message(message)

        # Print response
        print("Bot:", response)