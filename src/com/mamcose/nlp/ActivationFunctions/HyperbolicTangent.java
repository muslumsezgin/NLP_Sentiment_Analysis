package com.mamcose.nlp.ActivationFunctions;

public class HyperbolicTangent implements ActivationFunction {

    @Override
    public double execute(double y) {
        return ((Math.exp(y) - Math.exp(-y)) / (Math.exp(y) + Math.exp(-y)));
    }

    @Override
    public double derivative(double y) {
        return 1 / (0.25 * Math.pow(Math.exp(y) + Math.exp(-y), 2));
    }

}
