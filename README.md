# CL-Final-Project

Research question and data

The object of my final project is to provide an understanding about the representation of Helsinki as a travel destination. This is why I decided to conduct a topic modelling project where I try to identify themes that appear in texts of tourist brochures. After finding the themes, I want to analyze how do these themes fit together and what kind of story do they tell for visitors.

For this project, I am handling a dataset that contains 30 tourist brochures produced by the city of Helsinki between 1967-2008. The dataset is open accessible and can be downloaded from the Language bank of Finland. It contains also visual data, but this time I will analyze only the texts that are found in XML format. The brochures are in English.

Method

To prepare the data for topic modelling, I downloaded the XML files into OpenRefine. There I transformed all text to lowercase letters by common transform option. Then I removed the punctuation of sentences and replaced it with a space by using regular expressions. After this, I split the multi-valued cells so that each sell has only one word. The rest of the preprocessing is done with Python by support of a DH topic modelling tutorial and that is why I exported the project into JSON format with the Templating exporter function. However, later on it appeared that something went wrong while exporting the data, because my results do not correspond the data that I handled in OpenRefine.

To execute a topic model, I must further process the data. I started to write a Python code in Jupyter notebook. First, I imported few libraries that I will need. To recognize the unwanted words from data, I downloaded a Natural language tool kit stopwords. To lemmatize the words, I downloaded Spacy library and to constitute a topic model and word frequency, I downloaded Gensim library. By following a DH tutorial, I managed to create the topic model and the visualization of data. However, my results are not in line with my preprocessed data. There is somehow only one word that has survived through my process. Although, it is correctly lemmatized. This is why in my visualization there is only one theme found.  

To be able to analyze the data and the pipeline, I must resolve the problem of data conversion. It seems that the texts from the cells have not been exported from OpenRefine or then the data is not downloaded correctly to Python. This is why I couldn’t complete the grouping of topics.
