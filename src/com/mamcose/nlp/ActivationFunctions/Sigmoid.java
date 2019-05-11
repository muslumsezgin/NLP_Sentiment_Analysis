package com.mamcose.nlp.ActivationFunctions;

public class Sigmoid implements ActivationFunction {

    private static Sigmoid instance;

    private Sigmoid(){}

    @Override
    public double execute(double y) {
        return (1 / 1 + Math.exp(-y));
    }

    @Override
    public double derivative(double y) {
        return (y * (1-y));
    }

    public static Sigmoid getInstance(){
        if(instance == null){
            instance = new Sigmoid();
        }
        return instance;
    }
}
