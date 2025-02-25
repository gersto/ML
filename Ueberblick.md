# AI

## Machine Learing Algorithmen

* computional methods that allow computers to learn from and make decisions or predictions based on the data
* they do this by identifying patterns and relationships
* during a process called "training"
* an algorithm will adjust its parameters in order to minimize errors
* once the training is finished the algorithm can made predictions on data he has never seen before
* this is the core of artifical intelligence
* key algorithms
  * linear regression
  * decision trees
  * random forests
  * k-means
  * k-nearest neighbors
  * support vector machines (SVMS)
* python libraries
  * scikit-learn (https://scikit-learn.org/stable/)
  * numpy (https://numpy.org/)
  * pandas (https://pandas.pydata.org/)
  * matplotlib (https://matplotlib.org/)
  * seaborn (https://seaborn.pydata.org/)
  * squid-AI (https://squid.cloud/)
 
## Neural Networks

* are specific type of machine learning algorithm whose architecture is inspired by the function of the human brain
* can automatically extract and learn more complex features of raw data
* components of neural networks
  * layers (input, hidden, output)
  * neurons and activation functions
  * backpropagation (helps the robot learn from its mistakes by fixing them)
  * gradient descent (guides the robot to the right answer by taking small, careful steps)
* modules
  * pytorch (https://pytorch.org/)
  * tensorflow (https://www.tensorflow.org/)
  * keras (https://keras.io/)
* neuroevolution of augmenting topologies (NEAT) - (https://neat-python.readthedocs.io/en/latest/neat_overview.html)

## Computer Vision

* refers to doing image and video analysis and typically things like object detection and tracking, facial recognition, image segmentation, etc
* opencv python
* convolutional neural networks (we take an image an analyse it using machine learning algorithm)
* modules
  * opencv (https://opencv.org/)
  * skikit-image (https://scikit-image.org/)
  * pillow (https://pillow.readthedocs.io/en/stable/ ; https://python-pillow.org/)
 
## LLMs

* one of the most misunderstood forms of AI in terms of what they actually do and how you can use them
* are designed to understand and generate human language in a broad sense
* they are trained on tons of data like
  * books
  * movies
  * articles
  * internet
  * ...
* they can answer questions
* they can generate essays
* they can write code
* fine-tuning is the process of passing specific data related to exactly what you want this llm to do
  * medical diagnosis
  * ...
* key models
  * GPT (generative pre-trained transformers)
    * generates text, processes text unidirectionally (left and right) and is great for creating new content
  * BERT (bidirectional encoder representations from transformers)
    * understands text, processes text bidirectionally (left to right and right to left), and is great for understanding and interpreting existing content.
* python modules
  * huggingface (https://huggingface.co/)

## RAG (Retrieval augmented generation)

* technique that can be used with LLMs to get better responses and use them for more context-specific application
* LLMs typically don't have access to real-time information
* or even if they do, they might not have access to the information that you need them to have
* RAG involves the LLMs querying a specific type of dataset, grabbing information that's relevant to what it needs to answer, and then giving you a context-specific response
* taking the data that you want this model to have  access to, storing it in a really fast database to look up stuff from: vector search/store database.
* the LLM will then use that database by providing some kind of prompt, it will get relevant results back, it will read and understand that and then generate something that's context-specific
* python products
  * LangChain (https://www.langchain.com/)
  * Ollama (https://ollama.com/)
  * LlamaIndex (https://www.llamaindex.ai/)
 
## AI Agents

* they have the ability to interact with their environment and use various tools that you give them access to
* could very simple like having the ability to send an email
* or has access to full suite of different tools
* you can build a virtual assistent
* key components
  * environment interaction
  * self-determined task execution
  * data collection
* python products
  * LangChain (https://www.langchain.com/)
  * LlamaIndex (https://www.llamaindex.ai/)

### Link
[https://www.youtube.com/watch?v=OHf5bapbrcl](https://www.youtube.com/watch?v=OHf5bapbrcl)
