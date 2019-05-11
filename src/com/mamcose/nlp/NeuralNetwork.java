package com.mamcose.nlp;

import com.mamcose.nlp.ActivationFunctions.HyperbolicTangent;

import java.util.ArrayList;
import java.util.stream.IntStream;

public class NeuralNetwork {

    private double biasInput = 1.0;
    private double biasHidden = 1.0;
    private double learningRate = 0.08;

    private ArrayList<Neuron> inputLayer, hiddenLayer;
    private Neuron outputNeuron;

    public NeuralNetwork() {
        inputLayer = new ArrayList<>();
        hiddenLayer = new ArrayList<>();
    }

    public void init(int inputLayerSize, int hiddenLayerSize) {
        initLayers(inputLayer, inputLayerSize, biasInput);
        initLayers(hiddenLayer, hiddenLayerSize, biasHidden);
        outputNeuron = new Neuron(HyperbolicTangent.getInstance());
        initConnections();
    }

    /**
     * Katmanlara nöron ekleme işlemini yapar
     * @param list Katman
     * @param size Eklenecek nöron sayısı
     * @param bias Son nörona eklenecek bias değeri
     */
    private void initLayers(ArrayList<Neuron> list, int size, double bias) {
        IntStream.range(0, size - 1).forEach(i -> list.add(new Neuron()));
        list.add(new Neuron(bias));
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

            //double tweakedWeight = output * HyperbolicTangent.getInstance().derivative(error) * HyperbolicTangent.getInstance().derivative(forwardPropResult);
            //double tweakedWeight = error * (output) * (1 - Math.pow(forwardPropResult, 2));
            double tweakedWeight = error * (output) * (forwardPropResult) * (1 - forwardPropResult);

            double tweakedBias = error * (forwardPropResult)*(1-forwardPropResult);

            neuronToTweak.getConnections().get(i).updateWeight(learningRate * tweakedWeight);
            biasHidden -= tweakedBias * learningRate;
        }
        // Input layer ve hidden layer arasındaki bağlantıları ayarlıyoruz
        for (Neuron neuron : inputLayer) {
            for (int j = 0; j < hiddenLayer.size(); j++) {
                double deltaWeight = error * neuron.getOutputDerivative() * neuron.getOutput();
                double deltaBias = error*neuron.getOutputDerivative();
                neuron.getConnections().get(j).updateWeight(learningRate * deltaWeight);
                biasInput -= deltaBias * learningRate;
            }
        }
    }

    public void setBiasHidden(double biasHidden) {
        this.biasHidden = biasHidden;
    }

    public void setBiasInput(double biasInput) {
        this.biasInput = biasInput;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}
