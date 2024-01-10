# ELMo
This is an ELMo implementation from scratch using PyTorch. The core implementation involves adding a stacked Bi-LSTM - each of which gives the embedding for a word in a sentence, and a trainable parameter for weighing the word embeddings obtained at each layer of the ELMo network.

We train this ELMo architecture on a 4-way classification task using the AG News Classification Dataset. 

## Files
- 2-elmo.ipynb and 2-elmo.py contain the same code. 
- They contain the three main parts: ELMO architecture, Model Pretraining and Downstream Classification Task.
- You can view the ELMo Model from this link: https://drive.google.com/file/d/1-883UMV9h3Zk03-1jgLWFOhLEDFQaQzM/view?usp=sharing
- You can view the Sentence Classification model from this link: https://drive.google.com/file/d/1-2lwbZzv0wlwhrXaOgFFFRJzxNDPQbNx/view?usp=sharing

<br>
<br>

## Analysis
### Analysing the ELMo Model Pretraining
- Test Loss: -5376.0562
- Accuracy: 0.930
- Precision: 0.920
- Recall: 0.940
- F1 Score: 0.930

### Analysing the Downstream Task
- Test Accuracy: 0.78
- Test F1 Micro: 0.76

<br>
<br>


## Related Theory
## 3.1) Theory  
<br>

### 1) __ELMo vs CoVe__
- ELMo is based on a bidirectional LSTM (Long Short-Term Memory) language model. It leverages the idea that the meaning of a word can be highly dependent on its context within a sentence. ELMo addresses this by providing a contextually rich representation for each word in a sentence.  <br> <br>
  
- CoVe, on the other hand, combines word representations with representations from a neural machine translation (NMT) system. It's designed to capture context from both source and target sentences in a translation task.<br> <br>
  
- ELMo is pretrained on a bidirectional language modeling task, where it learns to predict both the next and previous words in a sentence, enabling it to capture bidirectional context. CoVe, on the other hand, is trained on a machine translation task, which involves translating English sentences to a foreign language and back. This task encourages CoVe to capture context from translation pairs. <br><br>
  
- ELMo uses a six-layer bidirectional LSTM stack, providing a deeper architecture to capture context, whereas CoVe uses a two-layer bidirectional LSTM, which is shallower. <br> <br>
  
- ELMo sums up the hidden states of each LSTM in its stack to obtain contextual embeddings, allowing for a richer representation that includes information from all layers. In contrast, CoVe takes the final outputs only, which means it doesn't explicitly consider the intermediate hidden states from the LSTM layers. <br> <br>
  
- CoVe is paired with a two-layer decoder LSTM during its pretraining, reflecting its machine translation context. In contrast, ELMo is pretrained without a specific paired architecture, and it's commonly used with a classification head to predict the next word when fine-tuned for various downstream tasks. <br> <br>

- Both ELMo and CoVe use global word embeddings (typically GloVe) to enhance their contextual embeddings. However, they differ in how they incorporate these global embeddings. ELMo simply adds them to the forward and backward embeddings, while CoVe concatenates its contextual representation with GloVe to create the final representation. This means that GloVe is more explicitly combined with CoVe's contextualized representations. <br> <br>


### 2) __Character Convolutional Layer__ 
- __Definition__
    - The character convolutional layer is essentially an application of Convolutional Neural Networks (CNNs) to sequences of characters, as opposed to the traditional use of CNNs on pixel data. 
  
    - It involves applying convolutional filters to input character sequences to capture local patterns.

- __Layer__:
    - The convolutional filters slide over the character sequence, similar to how image filters slide over pixel data. These filters are responsible for extracting features from the characters.

    - After applying the convolutional filters, max-pooling or other aggregation techniques are often used to obtain fixed-size representations for each character sequence.

    - The resulting character-level representations are then typically concatenated or combined in some way with word-level representations to create richer, context-aware word embeddings.

- __Purpose__:
    - __Subword Information__: The character convolutional layer allows the model to capture subword-level information. This is crucial because many languages have complex morphology and word formation rules, which can greatly affect word meaning.

    - __Out-of-Vocabulary Handling__: It helps in handling out-of-vocabulary words or rare words by providing subword-level information that may be common across different words.

    - __Robustness__: Subword information enhances the robustness of word embeddings by encoding structural and morphological details that can be particularly important in morphologically rich languages.

- __Alternatives__:
    - __Byte-Pair Encoding (BPE)__: BPE starts with characters as separate tokens and merges them iteratively based on their frequency until a predefined vocabulary size is reached. This results in subword tokens that can capture meaningful subword units and morphology.

    - __Morphological Analysis__: In languages with rich morphology, morphological analysis can be used to break words into their constituent morphemes, which are the smallest meaningful units of language. This approach can be an effective way to capture subword-level information.







