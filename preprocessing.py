import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import xml.etree.ElementTree as ET
import pandas as pd

ps = PorterStemmer()
lem = WordNetLemmatizer()


def get_xml_titles():
    # making list of xml filenames in xml folder
    path_to_xml_files = 'xml/'
    xml_filenames = [file for file in os.listdir(path_to_xml_files) if file.endswith('.xml')]

    return xml_filenames


def parse_xml():
    xml_filenames = get_xml_titles()
    # this will be the list of lists containing all rows (all words from all files)
    words = []

    for name in xml_filenames:
        root_node = ET.parse('xml/' + name).getroot()

        # store words in each file in their own row
        row = []

        for w in root_node.iter('WORD'):
            word = w.text
            word = word.lower()
            if word is not None and word.isalnum():
                row.append(word)

        # removing stop words from filtered words using nltk to ensure they do not count as most common.
        # words list w/out stopwords
        row = [lem.lemmatize(x) for x in row if x not in stopwords.words('english') and len(x) >= 3]

        words.append(row)

    return words


parse_xml()
