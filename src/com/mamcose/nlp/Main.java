package com.mamcose.nlp;

public class Main {

    public static void main(String[] args) {

        long startProcessTime = System.currentTimeMillis();

        int hiddenLayerSize = 10;
        int epochSize = 10;

        WordVectorizer wordVectorizer = new WordVectorizer();
        wordVectorizer.init();

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.init(wordVectorizer.getBagOfWords().size(), hiddenLayerSize);

        Logger.printInfo("Training Process");
        long startTime = System.currentTimeMillis();
        for (int j = 0; j < epochSize; j++) {
            for (int i = 0; i < wordVectorizer.getMatrix().size(); i++) {
                neuralNetwork.backPropagation(wordVectorizer.getMatrix().get(i), wordVectorizer.getLabels().get(i));
                Logger.printProgress(startTime, wordVectorizer.getMatrix().size() * epochSize, (j * wordVectorizer.getMatrix().size()) + (i + 1), "Epoch: " + (j + 1) + "/" + epochSize);
            }
        }

        Logger.printInfo("Test Process");
        startTime = System.currentTimeMillis();
        int count = 0;
        int sCount = 0;
        for (int i = 0; i < wordVectorizer.getMatrixTest().size(); i++) {
            double error = 0.5 * Math.pow(neuralNetwork.forwardPropagation(wordVectorizer.getMatrixTest().get(i)) - wordVectorizer.getLabelsTest().get(i), 2);
            if (error == 0.0) {
                count++;
            }
            if (error < 0.2) sCount++;
            Logger.printProgress(startTime, wordVectorizer.getMatrixTest().size(), i + 1, "");
        }

        Logger.printResult("Success Size: " + count);
        Logger.printResult(String.format("Accuracy: %.2f", count / (wordVectorizer.getMatrixTest().size() / 100.0)) + "%");
        Logger.printResult("Smooth Success Size: " + sCount);
        Logger.printResult(String.format("Smooth Accuracy: %.2f", sCount / (wordVectorizer.getMatrixTest().size() / 100.0)) + "%");
        Logger.printInfo("Total time: " + (System.currentTimeMillis() - startProcessTime) + "ms");

    }
}
