package com.mamcose.nlp;

import java.util.Collections;
import java.util.concurrent.TimeUnit;

public class Logger {

    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_BLACK = "\u001B[30m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_YELLOW = "\u001B[33m";
    public static final String ANSI_BLUE = "\u001B[34m";
    public static final String ANSI_PURPLE = "\u001B[35m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static final String ANSI_WHITE = "\u001B[37m";

    public static void printProgress(long startTime, long total, long current, String extra) {
        long eta = current == 0 ? 0 :
                (total - current) * (System.currentTimeMillis() - startTime) / current;

        String etaHms = current == 0 ? "N/A" :
                String.format("%02d:%02d:%02d", TimeUnit.MILLISECONDS.toHours(eta),
                        TimeUnit.MILLISECONDS.toMinutes(eta) % TimeUnit.HOURS.toMinutes(1),
                        TimeUnit.MILLISECONDS.toSeconds(eta) % TimeUnit.MINUTES.toSeconds(1));

        int percent = (int) (current * 100 / total);

        String string = '\r' +
                String.join("", Collections.nCopies(percent == 0 ? 2 : 2 - (int) (Math.log10(percent)), " ")) +
                String.format(" %d%% [", percent) +
                String.join("", Collections.nCopies(percent, "=")) +
                '>' +
                String.join("", Collections.nCopies(100 - percent, " ")) +
                ']' +
//                String.join("", Collections.nCopies((int) (Math.log10(total)) - (int) (Math.log10(current)), " ")) +
                String.format(" %d/%d, ETA: %s", current, total, etaHms) +
                "\t" + extra;
        System.out.print(string);
    }

    public static void print(String text) {
        System.out.print('\r' + text);
    }

    public static void printInfo(String text) {
        System.out.println("\n\n" + ANSI_BLUE + "INFO: " + text + ANSI_RESET);
    }

    public static void printResult(String text) {
        System.out.print("\n\n" + ANSI_GREEN + "RESULT: " + text + ANSI_RESET);
    }

    public static String printColor(String text, String color) {
        return color + text + ANSI_RESET;
    }
}
