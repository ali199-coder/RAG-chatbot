import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Simple knowledge base about BAT (Basic Attention Token)
knowledge_base = {
    "What is BAT?": "BAT (Basic Attention Token) is the utility token of the Brave browser ecosystem, designed for digital advertising and rewards.",
    "Who created BAT?": "BAT was created by Brendan Eich, the co-founder of Mozilla and creator of JavaScript.",
    "How does BAT work?": "BAT is used in the Brave browser to reward users for viewing ads and to compensate content creators.",
    "What is Brave browser?": "Brave is a privacy-focused web browser that blocks trackers and rewards users with BAT for viewing privacy-respecting ads.",
    "How can I earn BAT?": "You can earn BAT by enabling Brave Ads in the Brave browser or through content creation if you're a verified publisher.",
    "What can I do with BAT?": "You can tip content creators, exchange BAT for other currencies, or use it for premium services on supported platforms."
}

class RAGChatbot:
    def __init__(self):
        # Load a pre-trained sentence transformer model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_questions = list(knowledge_base.keys())
        self.knowledge_answers = list(knowledge_base.values())
        
        # Pre-compute embeddings for all questions in the knowledge base
        self.question_embeddings = self.model.encode(self.knowledge_questions)
    
    def find_most_relevant_answer(self, query):
        # Encode the query
        query_embedding = self.model.encode([query])
        
        # Compute similarity scores between query and all questions
        similarities = cosine_similarity(query_embedding, self.question_embeddings)
        most_similar_idx = np.argmax(similarities)
        
        # Get the most similar question and its answer
        most_similar_question = self.knowledge_questions[most_similar_idx]
        answer = knowledge_base[most_similar_question]
        
        return answer, similarities[0][most_similar_idx]
    
    def generate_response(self, query):
        answer, similarity_score = self.find_most_relevant_answer(query)
        
        # If the similarity is below a threshold, return a default response
        if similarity_score < 0.5:
            return "I'm not sure about that BAT-related question. Try asking about Basic Attention Token, Brave browser, or digital advertising rewards."
        
        return answer
    
    def chat(self):
        print("BAT Chatbot: Hello! Ask me anything about Basic Attention Token (BAT) or Brave browser.")
        print("Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("BAT Chatbot: Goodbye! Remember to browse with Brave and earn BAT!")
                break
            
            response = self.generate_response(user_input)
            print(f"BAT Chatbot: {response}")

# Create and run the chatbot
if __name__ == "__main__":
    chatbot = RAGChatbot()
    chatbot.chat()
