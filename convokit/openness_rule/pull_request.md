### Description

Feature: This PR introduces a new transformer that scores utterances with questions (0 otherwise).

The rule-based method checks to see if there are keywords associated with "open" or "closed" and adds or subtracts a point respectively. The list of keywords associated with questions is finite. This list includes the wh-questions previously explained through Pomerantz. For example, what, why, and how are all considered indicators that a question is "more open." While who, where, when, and which are all considered "more closed."

This takes from both the distinction that Robinson and Rackstraw make, but also Dohrenwend's idea that closed questions consist of the (1) selection question, (2) yes-no question, and (3) identification question.

Another notable category of indicator words are words that invoke opinion such as "wonder," "think," or "you." These tokens come from the human annotation tasks. We also relegate "yeah," "yes," and "no" as indicators of Dohrenwend's yes-no questions and rhetorical questions.

When checking against the utterance, we only score the last question. This is due to the fact that when asked a series of questions, respondents will usually answer the last one posed. This also helps us avoid scoring multiple questions in one utterance.

### Motivation and Context
We seek to examine the phenomena of question "openness" and how it might change throughout a conversation.

### How Has This Been Tested?
We ran sanity checks through identifying a few utterances in a corpus (CANDOR) and seeing if the score performs as expected one the transformer is applied.

### Other information:
N/A
