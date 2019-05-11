package com.mamcose.nlp.ActivationFunctions;

public class LeakyReLU implements ActivationFunction{

    private static LeakyReLU instance;

    private double leakiness;

    private LeakyReLU() {
        this.leakiness = 0.08;
    }

    private LeakyReLU(double leakiness) {
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

    public static LeakyReLU getInstance(){
        if(instance == null){
            instance = new LeakyReLU();
        }
        return instance;
    }

    public static LeakyReLU getInstance(double leakiness){
        if(instance == null){
            instance = new LeakyReLU(leakiness);
        }
        return instance;
    }

}
