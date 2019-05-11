package com.mamcose.nlp;

public class Main {

    public static void main(String[] args) {

        long startProcessTime = System.currentTimeMillis();

        int hiddenLayerSize = 50;
        int epochSize = 20;
        double tolerance = 0.2;
        double learningRate = 0.05;

        WordVectorizer wordVectorizer = new WordVectorizer();
        wordVectorizer.init();

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.setLearningRate(learningRate);
        neuralNetwork.init(wordVectorizer.getBagOfWords().size(), hiddenLayerSize);

        Logger.printInfo("Training Process");
        long startTime = System.currentTimeMillis();
        for (int j = 0; j < epochSize; j++) {
            for (int i = 0; i < wordVectorizer.getMatrix().size(); i++) {
                neuralNetwork.backPropagation(wordVectorizer.getMatrix().get(i), wordVectorizer.getLabels().get(i));
                Logger.printProgress(startTime, wordVectorizer.getMatrix().size() * epochSize,
                        (j * wordVectorizer.getMatrix().size()) + (i + 1), "Epoch: " + (j + 1) + "/" + epochSize);
            }
        }

        Logger.printInfo("Test Process");
        startTime = System.currentTimeMillis();
        int count = 0;
        int TP = 0,TN = 0,FP = 0,FN = 0;

        for (int i = 0; i < wordVectorizer.getMatrixTest().size(); i++) {
            double error = 0.5 * Math.pow(neuralNetwork.forwardPropagation(wordVectorizer.getMatrixTest().get(i)) - wordVectorizer.getLabelsTest().get(i), 2);

            if(wordVectorizer.getLabelsTest().get(i) == 1){
                if(error <= tolerance)
                    TP++;
                else
                    FN++;
            }else{
                if(error <= tolerance)
                    TN++;
                else
                    FP++;
            }

            if (error <= tolerance) count++;

            Logger.printProgress(startTime, wordVectorizer.getMatrixTest().size(), i + 1, "");
        }

        Logger.printResult("Success Size: " + count);
        Logger.printResult(String.format("Accuracy: %.2f", count / (wordVectorizer.getMatrixTest().size() / 100.0)) + "%");
        Logger.printResult("TP: " + TP);
        Logger.printResult("TN: " + TN);
        Logger.printResult("FP: " + FP);
        Logger.printResult("FN: " + FN);
        double precision = (double)TP / (TN + FN);
        double recall = (double) TP / (TP+FN);
        double fScore = (2.0 * TP) / (2.0 * TP + FP + FN);
        Logger.printResult("Precision: " + precision * 100 + "%");
        Logger.printResult("Recall: " + recall * 100 + "%");
        Logger.printResult("F-Score: " + fScore * 100 + "%");
        Logger.printInfo("Total time: " + Logger.printTime(startProcessTime));

    }
}
