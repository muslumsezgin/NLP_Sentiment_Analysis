package com.mamcose.nlp;

import java.util.ArrayList;

public interface Text2Matrix {
    String textCleaner(String line);
    void bagOfWordsCreator(String line);
    ArrayList<Integer> wordVectorizer(String line);
}
