package com.mamcose.nlp.ActivationFunctions;

public class ReLU implements ActivationFunction{

    private static ReLU instance;

    private ReLU(){}

    @Override
    public double execute(double y) {
        return y >= 0 ? y : 0;
    }

    @Override
    public double derivative(double y) {
        return y <= 0 ? 0 : 1;
    }

    public static ReLU getInstance(){
        if(instance == null){
            instance = new ReLU();
        }
        return instance;
    }

    public static ReLU getInstance(double leakiness){
        if(instance == null){
            instance = new ReLU();
        }
        return instance;
    }

}