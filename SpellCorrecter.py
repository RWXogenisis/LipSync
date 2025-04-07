import language_tool_python
import nltk
import pkg_resources
import re
from nltk.corpus import words
from nltk.metrics import edit_distance
from symspellpy import SymSpell, Verbosity

class Preprocessor:
    """Preprocessor class for initializing and loading NLP resources like spellcheckers and grammar tools."""
    
    def __init__(self):
        """
        Initializes the Preprocessor class by ensuring necessary NLP resources are downloaded.
        Loads SymSpell for spelling correction and LanguageTool for grammar correction.
        """
        # Ensure necessary NLTK data packages are downloaded
        self._safe_nltk_download("punkt", "tokenizers")
        self._safe_nltk_download("punkt_tab", "tokenizers")
        self._safe_nltk_download("words", "corpora")
        
        # Load the list of valid English words from NLTK
        try:
            self.valid_words = set(words.words())  # Set of valid English words
        except Exception as e:
            print(f"[WARNING] Failed to load NLTK words corpus: {e}")
            self.valid_words = {"place", "bin", "blue", "at", "again"}  # Default fallback words in case of failure

        # Initialize SymSpell for spelling correction
        self.sym_spell = SymSpell()

        # Load the dictionary and initialize LanguageTool for grammar correction (English)
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")  
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.tool = language_tool_python.LanguageTool("en-US")

    def _safe_nltk_download(self, package, category):
        """
        Ensures the necessary NLTK package is downloaded. If not, it will download it.
        
        Args:
            package (str): The NLTK package name.
            category (str): The category of the NLTK data (e.g., "tokenizers", "corpora").
        """
        try:
            nltk.data.find(f"{category}/{package}")
        except LookupError:
            nltk.download(package)

class SpellCorrector:
    """Handles spelling and grammar correction using the Preprocessor class."""
    
    def __init__(self, preprocessor: Preprocessor):
        """
        Initializes the SpellCorrector class with a preprocessor that contains resources 
        for spelling and grammar correction.
        
        Args:
            preprocessor (Preprocessor): The Preprocessor instance containing spellchecking and grammar tools.
        """
        self.preprocessor = preprocessor
    
    def is_number(self, word):
        """
        Checks if a word is a valid number (supports formats like 100000 or 1,00,000).
        
        Args:
            word (str): The word to be checked.
        
        Returns:
            bool: True if the word is a number, False otherwise.
        """
        return re.fullmatch(r"\d{1,3}(?:,\d{3})*|\d+", word) is not None  # Check if it is a number

    def is_valid_word(self, word):
        """
        Checks if a word exists in the list of valid English words from NLTK.
        
        Args:
            word (str): The word to be checked.
        
        Returns:
            bool: True if the word is valid, False otherwise.
        """
        return word.lower() in self.preprocessor.valid_words  # Check validity against the loaded words list

    def remove_repeated_letters(self, word):
        """
        Reduces excessive repeating letters in a word (e.g., 'happyyy' → 'hapy').
        
        Args:
            word (str): The word to be processed.
        
        Returns:
            str: The word with excessive repeated letters removed.
        """
        return re.sub(r'(.)\1{2,}', r'\1', word)  # Limit repeating letters to two occurrences

    def correct_word(self, word):
        """
        Corrects a single word using SymSpell while handling common errors like repeated letters.
        
        Args:
            word (str): The word to be corrected.
        
        Returns:
            str: The corrected word.
        """
        word = self.remove_repeated_letters(word)  # Handle excessive letter repetition first

        if self.is_number(word) or self.is_valid_word(word):  # If it is a number or a valid word, return it as is
            return word

        # Get suggestions for the word from SymSpell
        suggestions = self.preprocessor.sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)

        if not suggestions:  # If no suggestions, return the original word
            return word

        # Choose the best suggestion (with the smallest edit distance)
        best_match = min(suggestions, key=lambda s: edit_distance(word.lower(), s.term))
        corrected_word = best_match.term

        # Preserve original word's case
        if word[0].isupper():
            corrected_word = corrected_word.capitalize()

        return corrected_word

    def correct_sentence(self, sentence):
        """
        Corrects spelling and grammar in a sentence while keeping valid words.
        
        Args:
            sentence (str): The sentence to be corrected.
        
        Returns:
            str: The corrected sentence.
        """
        words_list = nltk.word_tokenize(sentence)  # Tokenize the sentence into words

        corrected_words = [self.correct_word(word) for word in words_list]  # Correct each word in the sentence
        corrected_sentence = " ".join(corrected_words)  # Join the words back into a corrected sentence

        # Use LanguageTool to fix grammar (only for complete sentences, not short fragments)
        if len(corrected_words) > 2:  # Assume fragments of less than 2 words do not need grammar correction
            final_correction = self.preprocessor.tool.correct(corrected_sentence)  # Grammar correction
        else:
            final_correction = corrected_sentence  # If it is a fragment, do not change the grammar

        return final_correction

# Example usage of the SpellCorrector
if __name__ == "__main__":
    """
    Example usage of the SpellCorrector class.
    Initializes Preprocessor and SpellCorrector, then corrects a sample sentence.
    """
    preprocessor = Preprocessor()
    corrector = SpellCorrector(preprocessor)

    # Correct spelling and grammar in the sentence
    sentence = "I amm veryy hungryyy buttt thee ressturant is closd"
    corrected = corrector.correct_sentence(sentence)

    print(f"Original: {sentence} \nCorrected: {corrected}")