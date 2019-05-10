package com.mamcose.nlp;

import com.mamcose.nlp.ActivationFunctions.HyperbolicTangent;

import java.util.ArrayList;
import java.util.stream.IntStream;

public class NeuralNetwork {

    private double biasInput = 0.0;
    private double biasHidden = 0.0;
    private double learning_rate = 0.08;

    private ArrayList<Neuron> inputLayer, hiddenLayer;
    private Neuron outputNeuron;

    public NeuralNetwork() {
        inputLayer = new ArrayList<>();
        hiddenLayer = new ArrayList<>();
    }

    public void init(int inputLayerSize, int hiddenLayerSize) {
        initLayers(inputLayer, inputLayerSize, biasInput);
        initLayers(hiddenLayer, hiddenLayerSize, biasHidden);
        outputNeuron = new Neuron(new HyperbolicTangent());
        initConnections();
    }

    private void initLayers(ArrayList<Neuron> list, int size, double bias) {
        IntStream.range(0, size - 1).forEach(i -> list.add(new Neuron()));
        list.add(new Neuron(bias));
    }

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
            inputLayer.get(i).setOutput(inputArray.get(i));
        }

        for (int i = 0; i < hiddenLayer.size() - 1; i++) {
            hiddenLayer.get(i).neuronOutputGuess();
        }

        outputNeuron.neuronOutputGuess();

        return outputNeuron.getOutput();
    }

    public double backPropagation(ArrayList<Integer> inputs, int actual) {

        double forwardPropResult = forwardPropagation(inputs); //first considers the net's output
        double error = (forwardPropResult - actual);

        //first we tweak the connections from hidden layer to output neuron
        for (int i = 0; i < hiddenLayer.size(); i++) {
            Neuron neuronToTweak = hiddenLayer.get(i); //we consider the connection of a neuron in hidden layer

            double output = neuronToTweak.getOutput();  // we get its result i.e. the input for the output layer

            //TODO hata fonk bias güncelle
            //multiplying by tanh derivative
            double tweakedWeight = error * (output) * (1 - Math.pow(forwardPropResult, 2));

            //double tweakedWeight =error*(output)*(forwardPropResult)*(1-forwardPropResult);
            //double tweakedBias = error* (forwardPropResult)*(1-forwardPropResult);
            neuronToTweak.getConnections().get(i).updateWeight(learning_rate * tweakedWeight);

        }


        // Now we adjust the connections between input layer and hidden layer

        for (Neuron neuron : inputLayer) {
            for (int j = 0; j < hiddenLayer.size(); j++) {

                //deltaWeight= amount by which to update weight

                double deltaWeight = error * neuron.getOutputDerivative() * neuron.getOutput();

                //TODO bias yap
                //double deltaBias = error*Neuron_Object.relu_deriv_func(neuron.output);

                neuron.getConnections().get(j).updateWeight(learning_rate * deltaWeight);


            }
        }


        return forwardPropResult;
    }


}
