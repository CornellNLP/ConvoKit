import re
import json
from sentence_transformers import SentenceTransformer, util
from evaluate import load

try:
    from convokit.genai import get_llm_client

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class ConDynSBaselines:
    """A class providing baseline methods for computing conversation similarity to compare with ConDynS.

    This class provides various baseline methods for comparing conversations including
    BERTScore, cosine similarity using sentence embeddings, and GPT-based comparison methods.
    The baseline methods are used to compare with ConDynS.

    :param model_provider: The GenAI provider to use (e.g., "gpt", "gemini")
    :param model: Optional specific model name
    :param sentence_transformer_model: Sentence transformer model to use for embeddings (default: "all-MiniLM-L6-v2")
    :param device: Device to use for sentence transformer (default: "cpu")
    """

    def __init__(
        self,
        model_provider: str,
        config,
        model: str = None,
        sentence_transformer_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """Initialize the ConDynSBaselines with specified models and configurations.

        :param model_provider: The GenAI provider to use (e.g., "gpt", "gemini")
        :param model: Optional specific model name
        :param sentence_transformer_model: Sentence transformer model to use for embeddings
        :param device: Device to use for sentence transformer
        :raises ImportError: If required dependencies are not available
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "GenAI dependencies not available. Please install via `pip install convokit[genai]`."
            )

        self.model_provider = model_provider
        self.model = model
        self.sentence_transformer_model = sentence_transformer_model
        self.device = device
        self.client = get_llm_client(model_provider, config, model=model)
        self.st_model = SentenceTransformer(sentence_transformer_model, device=device)
        self.util = util
        self.bertscore = load("bertscore")

    def get_bertscore(self, pred, ref):
        """Compute BERTScore between prediction and reference texts.

        Uses the BERTScore metric to evaluate semantic similarity between two texts.

        :param pred: Prediction text to evaluate
        :param ref: Reference text to compare against
        :return: BERTScore computation results
        """
        a = [pred]
        b = [ref]
        return self.bertscore.compute(
            predictions=a, references=b, model_type="distilbert-base-uncased"
        )

    def get_cosine_similarity(self, pred, ref):
        """Compute cosine similarity between two texts using sentence embeddings.

        Uses the SentenceTransformer model to generate embeddings and computes
        cosine similarity between them.

        :param pred: First text for comparison
        :param ref: Second text for comparison
        :return: Cosine similarity score between 0 and 1
        """
        embeddings = self.st_model.encode([pred, ref], convert_to_tensor=True)
        similarity = self.util.cos_sim(embeddings[0], embeddings[1])
        return similarity.item()

    def _parse_gpt_responses(self, response):
        """Parse and clean model responses containing JSON.

        Extracts JSON content from model responses that may contain markdown formatting
        and handles potential parsing errors.

        :param response: Raw response text from model
        :return: Parsed JSON data as dictionary
        """
        clean_json_str = re.sub(r"```json|```", "", response).strip()
        try:
            parsed_data = json.loads(clean_json_str)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for response: {response}")  # Debugging output if needed
        return parsed_data

    def get_gpt_compare_score(self, pred, ref, prompt):
        """Compare two texts using GPT model with a custom prompt.

        Sends a formatted prompt to GPT model to compare two texts and returns
        similarity score and reasoning.

        :param pred: First text for comparison
        :param ref: Second text for comparison
        :param prompt: Prompt template to use for comparison
        :return: Tuple of (similarity_score, reasoning)
        """
        gpt_prompt = prompt.format(pred=pred, ref=ref)
        response = self.client.generate(gpt_prompt)
        parsed_response = self._parse_gpt_responses(response)
        score = parsed_response["sim_score"]
        reason = parsed_response["reason"]
        return score, reason

    def get_naive_gpt_compare_score_SCDs(self, scd1, scd2):
        """Compare two Summary of Conversation Dynamics (SCD) using GPT.

        Compares two SCD summaries and rates their similarity based on persuasion
        trajectory and conversational dynamics, ignoring specific topics or claims.

        :param scd1: First SCD summary
        :param scd2: Second SCD summary
        :return: Tuple of (similarity_score, reasoning)
        """
        naive_gpt_compare_scd_prompt = """Compare the following two summary of conversation dynamics (SCD) of two online conversations, rate the similarity of the two conversations on a scale from 1 to 100, based on their persuasion trajectory reflected in the SCDs.
        
### **Key Aspects of Persuasion Trajectory**
- **Persuasion Strategies**: Logical reasoning, emotional appeals, rhetorical questions, citing authority, anecdotes, hypothetical scenarios, refuting counterarguments, shifting burden of proof, repetition, framing, social proof.
- **Interaction Dynamics**: Engagement patterns (e.g., single argument vs. back-and-forth), timing effects, persistence vs. resistance.
- **Response to Persuasion**: Agreement, concession, skepticism, counter-argument, disengagement, linguistic indicators (e.g., hedging, intensity, pronouns).
- **Trajectory of Persuasion**: Gradual shift, immediate agreement, persistent resistance, partial concession, reversal, stalemate.

### **Ignore**:
- **Do not consider specific topics, claims, or arguments.

### **Output Requirements**
Return a JSON object containing:
- `"sim_score"` (int): A similarity score between 0-100, representing how similar the conversations themselves are in **trajectory** based on the SCDs.
- `"reason"` (string, ≤30 words): A brief explanation of why the score was given, referencing key conversational dynamics.

### **Output Format (JSON)**
```json
{{
    "sim_score": <int>,
    "reason": "<brief explanation (≤30 words)>"
}}

### **Conversations**
Conversation 1 SCD:
{pred}

Conversation 2 SCD:
{ref}
"""
        score, reason = self.get_gpt_compare_score(scd1, scd2, naive_gpt_compare_scd_prompt)
        return score, reason

    def get_naive_gpt_compare_score_Transcripts(self, transcript1, transcript2):
        """Compare two conversation transcripts using GPT.

        Compares two conversation transcripts and rates their similarity based on
        conversational trajectory and dynamics, ignoring specific topics discussed.

        :param transcript1: First conversation transcript
        :param transcript2: Second conversation transcript
        :return: Tuple of (similarity_score, reasoning)
        """
        naive_gpt_compare_transcript_prompt = """Compare the following two online conversations and rate their similarity on a scale from 1 to 100, based on their trajectory.
        
### **Definition of Trajectory**
The trajectory of a conversation refers to its **dynamics**, including:
- **Changes in tone** (e.g., neutral to argumentative, formal to casual, sarcastic or sincere).
- **Patterns of interaction** (e.g., back-and-forth exchanges, long monologues, interruptions).
- **Conversation strategies** (e.g., persuasion, questioning, storytelling).
- **Order of the above trajectory events**

### **Ignore**:
- The topics discussed.
- Specific factual content.

### **Output Requirements**
Return a JSON object containing:
- `"sim_score"` (int): A similarity score between 0-100, representing how similar the conversations are in **trajectory**.
- `"reason"` (string, ≤30 words): A brief explanation of why the score was given, referencing key conversational dynamics.

### **Output Format (JSON)**
```json
{{
    "sim_score": <int>,
    "reason": "<brief explanation (≤30 words)>"
}}

### **Conversations**
Conversation 1:
{pred}

Conversation 2:
{ref}
"""
        score, reason = self.get_gpt_compare_score(
            transcript1, transcript2, naive_gpt_compare_transcript_prompt
        )
        return score, reason
