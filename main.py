import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from typing import Tuple, Optional, Dict, List

class BATChatbot:
    """An enhanced RAG-based chatbot for Basic Attention Token (BAT) and Brave browser information.
    
    Features:
    - Persistent knowledge base storage
    - Dynamic knowledge base updates
    - Better error handling
    - Configurable similarity threshold
    - Multiple response strategies
    - Conversation history
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.5,
                 knowledge_base_file: str = 'bat_knowledge.json'):
        """Initialize the BAT chatbot.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            similarity_threshold: Minimum similarity score to consider a match
            knowledge_base_file: Path to JSON file storing the knowledge base
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.knowledge_base_file = knowledge_base_file
        self.conversation_history = []
        
        # Load or initialize knowledge base
        if os.path.exists(knowledge_base_file):
            with open(knowledge_base_file, 'r') as f:
                self.knowledge_base = json.load(f)
        else:
            self.knowledge_base = {
                "What is BAT?": "BAT (Basic Attention Token) is the utility token of the Brave browser ecosystem, designed for digital advertising and rewards.",
                "Who created BAT?": "BAT was created by Brendan Eich, the co-founder of Mozilla and creator of JavaScript.",
                "How does BAT work?": "BAT is used in the Brave browser to reward users for viewing ads and to compensate content creators.",
                "What is Brave browser?": "Brave is a privacy-focused web browser that blocks trackers and rewards users with BAT for viewing privacy-respecting ads.",
                "How can I earn BAT?": "You can earn BAT by enabling Brave Ads in the Brave browser or through content creation if you're a verified publisher.",
                "What can I do with BAT?": "You can tip content creators, exchange BAT for other currencies, or use it for premium services on supported platforms."
            }
            self._save_knowledge_base()
        
        # Pre-compute embeddings
        self._update_embeddings()
    
    def _update_embeddings(self) -> None:
        """Update question embeddings when knowledge base changes."""
        self.knowledge_questions = list(self.knowledge_base.keys())
        self.question_embeddings = self.model.encode(self.knowledge_questions)
    
    def _save_knowledge_base(self) -> None:
        """Save the current knowledge base to file."""
        with open(self.knowledge_base_file, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
    
    def add_to_knowledge_base(self, question: str, answer: str) -> None:
        """Add a new Q&A pair to the knowledge base.
        
        Args:
            question: The question to add
            answer: The corresponding answer
        """
        self.knowledge_base[question] = answer
        self._save_knowledge_base()
        self._update_embeddings()
    
    def find_most_relevant_answer(self, query: str) -> Tuple[Optional[str], float]:
        """Find the most relevant answer for a query.
        
        Args:
            query: User's input question
            
        Returns:
            Tuple of (answer, similarity_score) or (None, 0.0) if no match found
        """
        try:
            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, self.question_embeddings)
            most_similar_idx = np.argmax(similarities)
            similarity_score = similarities[0][most_similar_idx]
            
            if similarity_score >= self.similarity_threshold:
                most_similar_question = self.knowledge_questions[most_similar_idx]
                return self.knowledge_base[most_similar_question], similarity_score
            return None, similarity_score
        except Exception as e:
            print(f"Error finding answer: {e}")
            return None, 0.0
    
    def generate_response(self, query: str) -> str:
        """Generate a response to the user's query.
        
        Args:
            query: User's input question
            
        Returns:
            Generated response
        """
        # Add to conversation history
        self.conversation_history.append(("user", query))
        
        answer, similarity_score = self.find_most_relevant_answer(query)
        
        if answer:
            response = answer
        else:
            response = (
                "I'm not sure about that BAT-related question. Try asking about:\n"
                "- Basic Attention Token (BAT)\n"
                "- Brave browser\n"
                "- Digital advertising rewards\n"
                "- BAT token utility\n\n"
                "You can also teach me by typing 'learn <question> | <answer>'"
            )
        
        # Add bot response to history
        self.conversation_history.append(("bot", response))
        return response
    
    def handle_special_commands(self, input_text: str) -> Optional[str]:
        """Handle special commands like 'learn' or 'history'.
        
        Args:
            input_text: User's input text
            
        Returns:
            Response if it's a special command, None otherwise
        """
        if input_text.lower().startswith('learn ') and '|' in input_text:
            try:
                _, rest = input_text.split('learn ', 1)
                question, answer = [part.strip() for part in rest.split('|', 1)]
                if question and answer:
                    self.add_to_knowledge_base(question, answer)
                    return "Thank you! I've learned something new about BAT."
            except ValueError:
                return "Please use format: learn <question> | <answer>"
        
        elif input_text.lower() == 'history':
            return "\n".join([f"{role}: {text}" for role, text in self.conversation_history[-10:]])
        
        elif input_text.lower() == 'help':
            return (
                "BAT Chatbot Commands:\n"
                "- ask any BAT-related question\n"
                "- 'learn <question> | <answer>' - teach me something new\n"
                "- 'history' - view recent conversation\n"
                "- 'help' - show this help message\n"
                "- 'quit' - exit the chatbot"
            )
        
        return None
    
    def chat(self) -> None:
        """Start the interactive chat session."""
        print("BAT Chatbot: Hello! Ask me anything about Basic Attention Token (BAT) or Brave browser.")
        print("Type 'help' for commands, or 'quit' to exit.")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("BAT Chatbot: Goodbye! Remember to browse with Brave and earn BAT!")
                    break
                
                # Check for special commands
                special_response = self.handle_special_commands(user_input)
                if special_response is not None:
                    print(f"BAT Chatbot: {special_response}")
                    continue
                
                # Generate and display response
                response = self.generate_response(user_input)
                print(f"BAT Chatbot: {response}")
                
            except KeyboardInterrupt:
                print("\nBAT Chatbot: Goodbye! Happy BAT earning!")
                break
            except Exception as e:
                print(f"BAT Chatbot: Sorry, I encountered an error: {e}")
                continue

if __name__ == "__main__":
    # Initialize with custom parameters
    chatbot = BATChatbot(
        model_name='all-MiniLM-L6-v2',
        similarity_threshold=0.45,
        knowledge_base_file='bat_knowledge.json'
    )
    chatbot.chat()
