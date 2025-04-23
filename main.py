import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from typing import Tuple, Optional, Dict, List
from datetime import datetime
import logging

class BATChatbot:
    """A robust RAG-based chatbot for Basic Attention Token (BAT) and Brave browser information.

    Features:
    - Persistent knowledge base with version control
    - Dynamic knowledge base updates with validation
    - Advanced error handling and logging
    - Configurable similarity threshold with fallback strategies
    - Conversation history with timestamp
    - Context-aware responses
    - Input validation and sanitization
    """

    DEFAULT_KNOWLEDGE_BASE = {
        "What is BAT?": "BAT (Basic Attention Token) is the utility token of the Brave browser ecosystem, designed for digital advertising and rewards.",
        "Who created BAT?": "BAT was created by Brendan Eich, the co-founder of Mozilla and creator of JavaScript.",
        "How does BAT work?": "BAT is used in the Brave browser to reward users for viewing ads and to compensate content creators.",
        "What is Brave browser?": "Brave is a privacy-focused web browser that blocks trackers and rewards users with BAT for viewing privacy-respecting ads.",
        "How can I earn BAT?": "You can earn BAT by enabling Brave Ads in the Brave browser or through content creation if you're a verified publisher.",
        "What can I do with BAT?": "You can tip content creators, exchange BAT for other currencies, or use it for premium services on supported platforms."
    }

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.5,
        knowledge_base_file: str = 'bat_knowledge.json',
        max_history: int = 50
    ):
        """Initialize the BAT chatbot with configurable parameters.

        Args:
            model_name: SentenceTransformer model name
            similarity_threshold: Minimum similarity score for matches
            knowledge_base_file: Path to JSON knowledge base
            max_history: Maximum number of conversation entries to store
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self.knowledge_base_file = knowledge_base_file
        self.max_history = max_history
        self.conversation_history: List[Tuple[str, str, str]] = []  # (role, text, timestamp)

        # Load or initialize knowledge base
        self.knowledge_base = self._load_knowledge_base()
        self._update_embeddings()

    def _load_knowledge_base(self) -> Dict[str, str]:
        """Load knowledge base from file or initialize with defaults."""
        try:
            if os.path.exists(self.knowledge_base_file):
                with open(self.knowledge_base_file, 'r') as f:
                    kb = json.load(f)
                    if not isinstance(kb, dict):
                        raise ValueError("Invalid knowledge base format")
                    return kb
            else:
                self.logger.info("Creating new knowledge base")
                self._save_knowledge_base(self.DEFAULT_KNOWLEDGE_BASE)
                return self.DEFAULT_KNOWLEDGE_BASE.copy()
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
            self._save_knowledge_base(self.DEFAULT_KNOWLEDGE_BASE)
            return self.DEFAULT_KNOWLEDGE_BASE.copy()

    def _save_knowledge_base(self, kb: Dict[str, str]) -> None:
        """Save knowledge base to file with backup."""
        try:
            # Create backup if file exists
            if os.path.exists(self.knowledge_base_file):
                backup_file = f"{self.knowledge_base_file}.bak"
                os.rename(self.knowledge_base_file, backup_file)

            with open(self.knowledge_base_file, 'w') as f:
                json.dump(kb, f, indent=2)
            self.logger.info("Knowledge base saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving knowledge base: {e}")

    def _update_embeddings(self) -> None:
        """Update question embeddings for the knowledge base."""
        try:
            self.knowledge_questions = list(self.knowledge_base.keys())
            if self.knowledge_questions:
                self.question_embeddings = self.model.encode(self.knowledge_questions, show_progress_bar=False)
            else:
                self.question_embeddings = np.array([])
            self.logger.info("Embeddings updated successfully")
        except Exception as e:
            self.logger.error(f"Error updating embeddings: {e}")

    def add_to_knowledge_base(self, question: str, answer: str) -> bool:
        """Add a new Q&A pair to the knowledge base with validation.

        Args:
            question: The question to add
            answer: The corresponding answer

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            question = question.strip()
            answer = answer.strip()
            if not question or not answer:
                self.logger.warning("Empty question or answer provided")
                return False
            if len(question) > 500 or len(answer) > 2000:
                self.logger.warning("Question or answer too long")
                return False

            self.knowledge_base[question] = answer
            self._save_knowledge_base(self.knowledge_base)
            self._update_embeddings()
            self.logger.info(f"Added new Q&A: {question}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding to knowledge base: {e}")
            return False

    def find_most_relevant_answer(self, query: str) -> Tuple[Optional[str], float]:
        """Find the most relevant answer for a query.

        Args:
            query: User's input question

        Returns:
            Tuple of (answer, similarity_score) or (None, 0.0) if no match
        """
        try:
            if not self.knowledge_questions:
                return None, 0.0

            query_embedding = self.model.encode([query], show_progress_bar=False)
            similarities = cosine_similarity(query_embedding, self.question_embeddings)
            most_similar_idx = np.argmax(similarities)
            similarity_score = similarities[0][most_similar_idx]

            if similarity_score >= self.similarity_threshold:
                return self.knowledge_base[self.knowledge_questions[most_similar_idx]], similarity_score
            return None, similarity_score
        except Exception as e:
            self.logger.error(f"Error finding relevant answer: {e}")
            return None, 0.0

    def generate_response(self, query: str) -> str:
        """Generate a context-aware response to the user's query.

        Args:
            query: User's input question

        Returns:
            Generated response
        """
        query = query.strip()
        if not query:
            return "Please provide a valid question."

        # Add to conversation history
        self.conversation_history.append(("user", query, datetime.now().isoformat()))
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

        answer, similarity_score = self.find_most_relevant_answer(query)

        if answer:
            response = answer
        else:
            response = (
                "I don't have specific information on that BAT-related question. Try asking about:\n"
                "- Basic Attention Token (BAT) functionality\n"
                "- Brave browser features\n"
                "- Earning and using BAT\n"
                "- Digital advertising rewards\n\n"
                "You can teach me using 'learn <question> | <answer>'"
            )

        self.conversation_history.append(("bot", response, datetime.now().isoformat()))
        return response

    def handle_special_commands(self, input_text: str) -> Optional[str]:
        """Handle special commands like 'learn', 'history', or 'help'.

        Args:
            input_text: User's input text

        Returns:
            Response if special command, None otherwise
        """
        input_text = input_text.strip().lower()
        
        if input_text.startswith('learn ') and '|' in input_text:
            try:
                _, rest = input_text.split('learn ', 1)
                question, answer = [part.strip() for part in rest.split('|', 1)]
                if self.add_to_knowledge_base(question, answer):
                    return "Successfully learned new BAT information!"
                return "Failed to learn. Please check the format and content."
            except ValueError:
                return "Please use format: learn <question> | <answer>"

        elif input_text == 'history':
            if not self.conversation_history:
                return "No conversation history available."
            return "\n".join([f"[{ts}] {role}: {text}" for role, text, ts in self.conversation_history[-10:]])

        elif input_text == 'help':
            return (
                "BAT Chatbot Commands:\n"
                "- Ask any BAT or Brave-related question\n"
                "- 'learn <question> | <answer>' - Teach new information\n"
                "- 'history' - View last 10 conversation entries\n"
                "- 'help' - Show this help message\n"
                "- 'quit' - Exit the chatbot"
            )

        return None

    def chat(self) -> None:
        """Start an interactive chat session with improved UX."""
        print("Welcome to BAT Chatbot! Ask about Basic Attention Token (BAT) or Brave browser.")
        print("Type 'help' for commands or 'quit' to exit.")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == 'quit':
                    print("BAT Chatbot: Thanks for chatting! Keep exploring with Brave!")
                    break

                special_response = self.handle_special_commands(user_input)
                if special_response:
                    print(f"BAT Chatbot: {special_response}")
                    continue

                response = self.generate_response(user_input)
                print(f"BAT Chatbot: {response}")

            except KeyboardInterrupt:
                print("\nBAT Chatbot: Thanks for chatting! Stay private with Brave!")
                break
            except Exception as e:
                self.logger.error(f"Chat error: {e}")
                print("BAT Chatbot: Sorry, something went wrong. Please try again.")

if __name__ == "__main__":
    chatbot = BATChatbot(
        model_name='all-MiniLM-L6-v2',
        similarity_threshold=0.45,
        knowledge_base_file='bat_knowledge.json',
        max_history=100
    )
    chatbot.chat()
