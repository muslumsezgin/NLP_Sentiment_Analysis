package com.mamcose.nlp;

import com.mamcose.nlp.ActivationFunctions.HyperbolicTangent;

import java.util.ArrayList;
import java.util.stream.IntStream;

public class NeuralNetwork {

    private double learningRate = 0.25;

    private ArrayList<Neuron> inputLayer, hiddenLayer;
    private Neuron outputNeuron;

    public NeuralNetwork() {
        inputLayer = new ArrayList<>();
        hiddenLayer = new ArrayList<>();
    }

    public void init(int inputLayerSize, int hiddenLayerSize) {
        initLayers(inputLayer, inputLayerSize);
        initLayers(hiddenLayer, hiddenLayerSize);
        outputNeuron = new Neuron(HyperbolicTangent.getInstance());
        initConnections();
    }

    /**
     * Katmanlara nöron ekleme işlemini yapar
     * @param list Katman
     * @param size Eklenecek nöron sayısı
     */
    private void initLayers(ArrayList<Neuron> list, int size) {
        IntStream.range(0, size).forEach(i -> list.add(new Neuron()));
    }

    /**
     * Nöronlar arasındaki bağlantıları kurar
     */
    private void initConnections() {
        for (Neuron i : inputLayer) {
            for (Neuron h : hiddenLayer) {
                Connections c = new Connections(i, h);
                i.addConnections(c);
                h.addConnections(c);
            }
        }

        for (Neuron h : hiddenLayer) {
            Connections c = new Connections(h, outputNeuron);
            h.addConnections(c);
            outputNeuron.addConnections(c);
        }

    }

    public double forwardPropagation(ArrayList<Integer> inputArray) {
        for (int i = 0; i < inputArray.size(); i++) {
            inputLayer.get(i).setOutput(inputArray.get(i)); // input layerdaki nöronların çıkışlarını set eder
        }
        for (int i = 0; i < hiddenLayer.size() - 1; i++) {
            hiddenLayer.get(i).neuronOutputGuess(); // hidden layerdaki nöronların çıkışlarını hesaplar
        }
        outputNeuron.neuronOutputGuess(); // output nöronunun çıkışını hesaplar
        return outputNeuron.getOutput(); // output nöronunun çıkışını döner
    }

    public void backPropagation(ArrayList<Integer> inputs, int actual) {
        double forwardPropResult = forwardPropagation(inputs); // output nöronunun çıkış değeri
        double error = (forwardPropResult - actual);

        for (int i = 0; i < hiddenLayer.size(); i++) {
            Neuron neuronToTweak = hiddenLayer.get(i);
            double output = neuronToTweak.getOutput();

            double tweakedWeight = error * (output) * (1 - Math.pow(forwardPropResult, 2));
            //double tweakedWeight = error * (output) * (forwardPropResult) * (1 - forwardPropResult);

            //double tweakedBias = neuronToTweak.getBias() * error * (1 - Math.pow(forwardPropResult, 2));
            double tweakedBias = neuronToTweak.getBias() * error * (forwardPropResult)*(1-forwardPropResult);

            outputNeuron.getConnections().get(i).updateWeight(learningRate * tweakedWeight);
            neuronToTweak.updateBias(learningRate * tweakedBias);
        }
        // Input layer ve hidden layer arasındaki bağlantıları ayarlıyoruz
        for (Neuron neuron : inputLayer) {
            for (int j = 0; j < hiddenLayer.size(); j++) {
                double deltaWeight = error * neuron.getOutputDerivative() * neuron.getOutput();
                double deltaBias = error * neuron.getOutputDerivative() * neuron.getBias();
                neuron.getConnections().get(j).updateWeight(learningRate * deltaWeight);
                neuron.updateBias(learningRate * deltaBias);
            }
        }
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}
