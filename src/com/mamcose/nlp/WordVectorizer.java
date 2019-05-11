package com.mamcose.nlp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class WordVectorizer implements Text2Matrix {
    private int indexData = 5;
    private int indexLabel = 0;

    private String pathTraining;
    private String pathTest;
    private String pathStopWords;

    private ArrayList<Integer> labels = new ArrayList<>();// eğitim verilerin etiketlerini tutar
    private ArrayList<Integer> labelsTest = new ArrayList<>();// test verilerinin etiketlerini tutar
    private ArrayList<ArrayList<Integer>> matrix = new ArrayList<>();// eğitim verisi kelime vektorlerinin matris hali
    private ArrayList<ArrayList<Integer>> matrixTest = new ArrayList<>();// test verisi kelime vektorlerinin matris hali

    private ArrayList<String> wordsList = new ArrayList<>(); //tüm kelimeler
    private ArrayList<String> stopWords = new ArrayList<>(); //etkisiz kelimeler
    private Set<String> bagOfWords = new HashSet<>(); //tekil tüm kelimeler

    private ArrayList<String> lines = new ArrayList<>();//eğitim verisindeki tüm cümleler
    private ArrayList<String> linesTest = new ArrayList<>();//test verisindeki tüm cümleler

    public WordVectorizer() {
        String os = System.getProperty("os.name").toLowerCase();
        if(os.indexOf("mac") >= 0){
            this.pathTraining = "./resources/training.csv";
            this.pathTest = "./resources/test.csv";
            this.pathStopWords = "./resources/stopwords.txt";
        }else if(os.indexOf("win") >= 0){
            this.pathTraining = ".\\resources\\training.csv";
            this.pathTest = ".\\resources\\test.csv";
            this.pathStopWords = ".\\resources\\stopwords.txt";
        }
    }

    public WordVectorizer(String pathTraining, String pathTest, String pathStopWords, int indexData, int indexLabel) {
        this.pathTraining = pathTraining;
        this.pathTest = pathTest;
        this.pathStopWords = pathStopWords;
        this.indexData = indexData;
        this.indexLabel = indexLabel;
    }

    public void init() {
        stopWordsReader();
        trainDataReader();
        testDataReader();
    }

    /**
     * Eğitim verilerini dosyadan okur ve listeye atar
     */
    private void trainDataReader() {
        File csv = new File(pathTraining);
        try (FileReader fr = new FileReader(csv); BufferedReader bfr = new BufferedReader(fr)) {
            Logger.printInfo("Reading training data");
            for (String line; (line = bfr.readLine()) != null; ) {
                String[] split = line.split(",");
                String data = textCleaner(split[indexData].toLowerCase());
                int label = Integer.parseInt(split[indexLabel].toLowerCase());

                lines.add(data);
                labels.add(label);
                bagOfWordsCreator(data);
                Logger.print("Reading size: " + lines.size());
            }
            removeLowFreqWords(); // düşük frekanslı kelimeleri temizler

            for (String l : lines) {
                matrix.add(wordVectorizer(l));
            }
        } catch (IOException e) {
            Logger.printError("!! Training Data Not Reading !!");
            System.exit(0);
        }
    }

    /**
     * Test verilerini dosyadan okur ve listeye atar
     */
    private void testDataReader() {
        File csv = new File(pathTest);
        try (FileReader fr = new FileReader(csv); BufferedReader bfr = new BufferedReader(fr)) {
            Logger.printInfo("Reading test data");
            for (String line; (line = bfr.readLine()) != null; ) {
                String[] split = line.split(",");
                String data = textCleaner(split[indexData].toLowerCase());
                int label = Integer.parseInt(split[indexLabel].toLowerCase());

                linesTest.add(data);
                labelsTest.add(label);
                Logger.print("Reading size: " + linesTest.size());
            }
            for (String l : linesTest) {
                matrixTest.add(wordVectorizer(l));
            }
        } catch (IOException ex) {
            Logger.printError("!! Test Data Not Reading !!");
            System.exit(0);
        }
    }

    /**
     * Gereksiz kelimeler dosyasını okur ve listeye atar
     */
    private void stopWordsReader() {
        try {
            File file = new File(pathStopWords);
            FileReader fr = new FileReader(file);
            BufferedReader bfr = new BufferedReader(fr);
            bfr.lines().forEach(stopWord -> this.stopWords.add(stopWord));
            bfr.close();
        } catch (IOException e) {
            Logger.printError("!! Stop Words Not Found !!");
            System.exit(0);
        }
    }

    /**
     * bagOfWords deki düşük frekanslı kelimeleri temizler.
     */
    private void removeLowFreqWords() {
        Logger.printInfo("Remove Low Frequency Words");
        long startTime = System.currentTimeMillis();
        Set<String> set = new HashSet<>(wordsList);
        int i = 1;
        for (String s : set){
            if ((Collections.frequency(wordsList, s) <= 2) || " ".equals(s) || "".equals(s)) {
                bagOfWords.remove(s);
            }
            Logger.printProgress(startTime, set.size(),i++,"");
        }
        bagOfWords.removeAll(stopWords);
    }

    /**
     * Cümledeki gereksiz verileri temizler. (Örn: url, noktalama işaretleri, rakamlar)
     *
     * @param line cümle
     * @return temizlenmiş cümle
     */
    @Override
    public String textCleaner(String line) {
        return line.replaceAll("[!#$%^&*?|'_=\\[\\],.;?\"0-9/;():-]", "")
                .replace("RT", "")
                .replaceAll("http.*?\\s", "")
                .replaceAll("@.*?\\s", "")
                .replaceAll("www.*?\\s", "")
                .replace("quot", "")
                .replace("amp", "");
    }

    /**
     * Cümleyi kelimelere ayırır ve wordList ve bagOfWords e bu kelimeleri ekler.
     *
     * @param line cümle
     */
    @Override
    public void bagOfWordsCreator(String line) {
        String[] tokens = line.trim().split("\\s+");
        for (String token : tokens) {
            bagOfWords.add(token);
            wordsList.add(token);
        }
    }

    /**
     * Nöron ağını besleyecek şekilde cümleyi vektöre dönüştürür.
     * Kelime bagOfWords de varsa vektörde 1 ile temsil edilir. Yoksa 0 ile temsil edilir.
     *
     * @param line cümle
     * @return kelime vektörü
     */
    @Override
    public ArrayList<Integer> wordVectorizer(String line) {
        String[] data = line.split("\\s+");
        ArrayList<Integer> vector = new ArrayList<>(Collections.nCopies(bagOfWords.size(), 0));//bagOfWords boyutu kadar vektor oluştur ve içini 0 ile doldur.

        for (String s : data) {
            if (bagOfWords.contains(s)) {
                vector.set(new ArrayList<>(bagOfWords).indexOf(s), 1);
            }
        }
        return vector;
    }

    public ArrayList<Integer> getLabels() {
        return labels;
    }

    public ArrayList<Integer> getLabelsTest() {
        return labelsTest;
    }

    public ArrayList<ArrayList<Integer>> getMatrix() {
        return matrix;
    }

    public ArrayList<ArrayList<Integer>> getMatrixTest() {
        return matrixTest;
    }

    public Set<String> getBagOfWords() {
        return bagOfWords;
    }
}
