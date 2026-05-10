import json,os,nltk,random , torch
import numpy as np
import torch.nn as nn
from  torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
from nltk.stem import WordNetLemmatizer


# nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel,self).__init__()
        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3= nn.Linear(64,output_size)
        self.relu =nn.ReLU()
        self.dropout= nn.Dropout(0.5)



    def forward(self,x):
            x= self.relu(self.fc1(x))
            x =self.dropout(x)
            x= self.relu(self.fc2(x))
            x =self.dropout(x)
            x= self.fc3(x)
            return x

class ChatBotAssistant():
    def __init__(self,intents_paths,function_mapping=None):
        self.model =None
        self.intents_path = intents_paths
        self.documents=[]
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mapping = function_mapping

        self.X = None
        self.y = None

# to tokenize and lammatize
    @staticmethod
    def token_and_lam (text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)

        words = [lemmatizer.lemmatize(word.lower(), pos='v') for word in words]
        return words


# chatbot = ChatBotAssistant('responseplusquestions.json')
# print(chatbot.token_and_lam('giver givers giving givings'))
# # print(chatbot)


    
    def word_bags(self, words):
        return [ 1 if word in words else 0 for word in self.vocabulary]
    
    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer
        if os.path.exists(self.intents_path):
            with open(self.intents_path , 'r') as f:
                intents_data=json.load(f)

               

                for intent in intents_data['intents']:
                    if intent['tag'] not in self.intents:
                     self.intents.append(intent['tag'])
                     self.intents_responses[intent['tag']] = intent['responses']

                     for pattern in intent['patterns']:
                         pattern_words = self.token_and_lam(pattern)
                         self.vocabulary.extend(pattern_words)
                         self.documents.append((pattern_words,intent['tag']))

                     self.vocabulary = sorted(set(self.vocabulary))


    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words= document[0]
            bag= self.word_bags(words)

            intent_index = self.intents.index(document[1])


            bags.append(bag)
            indices.append(intent_index)

        self.X =np.array(bags)
        self.y =np.array(indices)
# train data


    def train_data(self,batch_size,lr,epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor =  torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset , batch_size=batch_size , shuffle=True)
        
        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss =0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output=self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss
            print(f"Epoch {epoch+1}: Loss {running_loss/len(loader):.4f}")
            # print(f"Epoch {epoch+1}: Loss {running_loss/len(loader):. 4f} ")


    def save_my_model(self, model_path, dimension_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimension_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size':len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open (dimensions_path, 'r') as f:
            dimensions= json.load(f)

            self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
            self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def to_process_message(self, input_messages):
        words =self.token_and_lam(input_messages)
        bags = self.word_bags(input_messages)

        bag_tensor = torch.tensor([bags], dtype= torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
            probabilities = F.softmax(predictions, dim =1)
            confidence = torch.max(
                    probabilities
                ).item()
                        
        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]
        if confidence < 0.70:
             return "I don't understand that yet."

        if self.function_mapping :
            if predicted_intent in self.function_mapping:
               return  self.function_mapping[predicted_intent]()


        if self.intents_responses[predicted_intent]:
            # return random.choice(self.intents_responses)
            return random.choice(self.intents_responses[predicted_intent])
        else:
            None



def get_stocks():
    stocks = ['META','APPL','MSFT','NVDA','GS']
    return random.sample (stocks,3)

if __name__ =='__main__':
    # assistant = ChatBotAssistant('responseplusquestions.json', function_mapping= {'stocks':get_stocks})
    # assistant.parse_intents()
    # assistant.prepare_data()
    # assistant.train_data (batch_size =8, lr=0.001, epochs =100) 

    # assistant.save_my_model('chatbot_model.pth','dimesions.json')

    assistant = ChatBotAssistant('responseplusquestions.json', function_mapping= {'stocks':get_stocks})
    assistant.parse_intents()
    assistant.load_model('chatbot_model.pth','dimesions.json')



    while True:

        message = input("You: ")

        if message.lower() == "quit":
            break

        response = assistant.to_process_message(message)

        print("Bot:", response)