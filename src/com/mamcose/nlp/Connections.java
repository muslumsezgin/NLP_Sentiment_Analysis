package com.mamcose.nlp;

import java.util.Random;

public class Connections {
    private Neuron from;
    private Neuron to;
    private double weight;

    public Connections(Neuron source, Neuron dest) {
        this.from = source;
        this.to = dest;
        weight = (new Random().nextDouble() - 0.45);
        if (weight == 0.0) weight = 0.1;
    }

    public double updateWeight(double update) {
        return weight -= update;
    }

    public double getWeight() {
        return weight;
    }

    public Neuron getFrom() {
        return from;
    }

    public Neuron getTo() {
        return to;
    }
}
