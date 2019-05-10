package com.mamcose.nlp.ActivationFunctions;

public class LeakyReLU implements ActivationFunction{

    private double leakiness;

    public LeakyReLU() {
        this.leakiness = 0.08;
    }

    public LeakyReLU(double leakiness) {
        this.leakiness = leakiness;
    }

    @Override
    public double execute(double y) {
        return y >= 0 ? y : y*leakiness;
    }

    @Override
    public double derivative(double y) {
        return y <= 0 ? 0 : 1;
    }
}
