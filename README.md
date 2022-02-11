Research question and data

The object of my final project is to provide an understanding about the representation of Helsinki as a travel destination for English speaking tourists. Therefore, I decided to conduct a LDA topic modelling project where I try to identify themes that appear in texts of tourist brochures. After finding the themes, I want to analyze how do these themes fit together and what kind of story do they tell for visitors.

For this project, I am handling a dataset that contains 30 tourist brochures produced by the city of Helsinki between 1967-2008. The dataset is accessible and can be downloaded from the Language bank of Finland. It contains also visual data, but this time I will analyze only the texts that are found in XML format. The brochures are in English.

Method

To prepare the data for topic modelling, I downloaded the XML files into OpenRefine. There I transformed all the text to lowercase letters by common transform option. Then I removed the punctuation of sentences and replaced it with a space. After this, I split the multi-valued cells that contained multiple words so that each cell had only one word. In this phase, I decided to remove all numbers from my data because I thought that those would be difficult to analyze without their specific context. The rest of the preprocessing was done in Python with several tools.

To execute a topic model, I continued processing the data further. I started to write a Python code in Jupyter notebook. First, I imported few libraries that are needed for processing the data and for building and visualizing topic model with Genism. To recognize the common words from data, I downloaded a Natural language tool kit stopwords. To lemmatize the words, I downloaded Spacy library and to constitute a topic model, I downloaded Gensim library. After preprocessing the data, I created the topic model with Gensim. I knew that the number of topics should be aligned with my data to have good results. I tried few different options and it seemed that thirteen topics would be somehow reasonable amount for my project. When I raised the number of topics, it became harder to understand what the relation of words inside the topic was. I set the program to present eight most relevant terms of each topic. Finally, I made visualization of my results with pyLDAvis tool.

Results

After I had built up the program that constituted the topics, I started to carefully study the terms inside them. My object was to find a common factor which would describe all the terms inside each topic and therefore be the title of that topic. Identifying the meaning of topic was easier to do for some topics than for others. For example, the lemmatization made some words very difficult to understand even I knew the context of the texts. Therefore, I had to do also close reading to be able to name some topics. The close reading revealed for example that in the topic 13, the world “glitter” was originally an attribute “glittering” for the word “sea”. Also, without close reading the brochure from 1972, where the title was “Come to Helsinki and relax”, it would be difficult to draw a line between words like “casserole”, “saleswoman”, “glitter” and “amusement”. They were used to describe a day that starts in the market square and ends up visiting the Linnanmäki amusement park. Despite of close reading, some topics were too abstract to identify.

I also used the relevance metric tool that counted the relevancy of terms inside the topics. This tool made it easy to compare which terms appeared frequently within the topic simply by sliding the cursor. It helped me to identify the precis meaning of the topic. For example, the topic 1 seemed to contain quite random selection of words like “restaurant”, “open”, “ finnish”, “museum” and “architecture”.  When I slided the cursor to see the relevant terms, I got “wheelchair”, “disabled” and “young”. I estimate that this topic is related to accessibility of different places in the city.

The topics that I identified with the method and the relevant terms:

1.	Accessibility (www, hel, fi, young, wheelchair, disabled, band, listen, architecture, drink)
2.	Environment (hectare, mill, headland, sun, necessary, interval, maintain, ply, rocky, beach)
3.	Infrastructure (lower, horizontal, create, fact, vertical, figure, peninsula, line, square, stone)
4.	Paid attractions (tel, admission, free, child, adult, weekday, building, group, appointment)
5.	Islands of Helsinki (motorboat, earthwork, bridge, connect, midsummer, suitable, tradition, summer, island, booth, museum, fortress)
6.	Timetables and tickets for boats/ships (line, pm, am, daily, open, photo, island, sight, discount, ship, purchase, nation)
7.	unknown (smile, factory, accessible, opening, exception, symbol, illustration, fee)
8.	Architecture (wikstr, kirkkopuisto, completion, puistikko, unknown, vanha, uniform, neoclassical) 
9.	unknown (reference, cross, start, end, view, address)
10.	Travelling to/from Helsinki (duration, mk, silja, hour, archipelago, booking, terminal, fee, line)
11.	unknown (hrs, photo, centre, face, hr, away, orchestra)
12.	unknown (social, program, characterize, glitter, foundation, overview)
13.	Relaxing day in Helsinki (foundation, social, saleswomen, characterize, glitter, program, casserole, amusement)

If I look at all the topics together, I can see that there is repetition of vocabulary concerning the life at the coast and the architecture. Also, I could identify these themes while I was close reading the brochures. For example, there were eight brochures that only talk about the Helsinki’s islands (Suomenlinna, Pihlajasaari, Seurasaari and Korkeasaari). It is no wonder that these texts contain a lot of information about moving around by boats and ships. Because of a strong emphasis on maritime vocabulary, the topics boost the representation of Helsinki as a “daughter of the Baltic Sea”. However, these topics are still quite incoherent and their interpretation requires simplifying.

I think that this project should be improved to get more reliable results. Firstly, there should be paid more attention for cleaning the data. The results contained words like “www”, “am”, “pm” or “fi” because they are used frequently to give information for tourists. It would be interesting to see if removing these kinds of words would give more descriptive results. Secondly, the program should be fixed what comes to reading Finnish words with “ä” or “ö”. Now these words are split and they have become unidentified. For example, the name of architect Emil Wikström has become “wikstr”. Thirdly, the different spellings of words should be considered. For example, word “motor-boat” is used in brochures in 1967 and 1976 while in 1984 it is spelled “motorboat”. 
