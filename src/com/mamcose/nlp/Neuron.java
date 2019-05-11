package com.mamcose.nlp;

import com.mamcose.nlp.ActivationFunctions.ActivationFunction;
import com.mamcose.nlp.ActivationFunctions.LeakyReLU;
import com.mamcose.nlp.ActivationFunctions.Sigmoid;

import java.util.ArrayList;

public class Neuron {

    private double output; //nöronun çıktısı
    private double bias = 1.0;
    private ArrayList<Connections> synapses; //nörondaki tüm bağlantılar
    private ActivationFunction activationFunction; //nöron aktivasyon fonksiyonu

    public Neuron() {
        output = 0;
        synapses = new ArrayList<>();
        activationFunction = Sigmoid.getInstance();
    }

    public Neuron(double output) {
        this.output = output;
        synapses = new ArrayList<>();
        activationFunction = Sigmoid.getInstance();
    }

    public Neuron(ActivationFunction activationFunction) {
        this.output = 0;
        synapses = new ArrayList<>();
        this.activationFunction = activationFunction;
    }

    public Neuron(double output, ActivationFunction activationFunction) {
        this.output = output;
        synapses = new ArrayList<>();
        this.activationFunction = activationFunction;
    }

    /**
     * Nöronun çıktısını günceller
     */
    public void neuronOutputGuess() {
        double sum = 0.0;
        for (Connections c : synapses) {
            if (this == c.getTo()) {
                sum += (c.getWeight() * c.getFrom().getOutput());
            }
        }
        output = activationFunction.execute(sum + bias);
    }

    /**
     * Nöronun bağlantılarını çevirir
     *
     * @return nöronun bağlantıları
     */
    ArrayList<Connections> getConnections() {
        return synapses;
    }

    /**
     * Gelen bağlantıyı nörunun bağlantılarına ekler
     *
     * @param conn gelen yeni bağlantı
     */
    void addConnections(Connections conn) {
        synapses.add(conn);
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getOutputDerivative() {
        return activationFunction.derivative(getOutput());
    }

    public double updateBias(double update){
        return bias -= update;
    }

    public double getBias() {
        return bias;
    }
}
