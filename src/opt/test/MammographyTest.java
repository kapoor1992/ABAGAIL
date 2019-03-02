package opt.test;

import opt.*;
import opt.example.*;   
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

public class MammographyTest {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 15, hiddenLayer = 8, outputLayer = 1, maxTrainingIterations = 500;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        run_small_iter();
        run_large_iter();
    }

    public static void run_small_iter() {
        double rhc_score[] = new double[4];
        double sa_score[] = new double[4];
        double ga_score[] = new double[4];

        for (int repeats = 0; repeats < 10; repeats++) {
            int[] trainingIterationsList = new int[] {1, 5, 10, 15};

            for(int trainingIterations : trainingIterationsList) {
                int ii = 0;

                if (trainingIterations > 1)
                    ii = trainingIterations / 5;

                for(int i = 0; i < oa.length; i++) {
                    networks[i] = factory.createClassificationNetwork(
                        new int[] {inputLayer, hiddenLayer, outputLayer});
                    nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
                }
        
                oa[0] = new RandomizedHillClimbing(nnop[0]);
                oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
                oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

                

                for(int i = 0; i < oa.length; i++) {
                    double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                    train(oa[i], networks[i], oaNames[i], trainingIterations); //trainer.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    trainingTime /= Math.pow(10,9);

                    Instance optimalInstance = oa[i].getOptimal();
                    networks[i].setWeights(optimalInstance.getData());

                    double predicted, actual;
                    start = System.nanoTime();
                    for(int j = 0; j < instances.length; j++) {
                        networks[i].setInputValues(instances[j].getData());
                        networks[i].run();

                        predicted = Double.parseDouble(instances[j].getLabel().toString());
                        actual = Double.parseDouble(networks[i].getOutputValues().toString());

                        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                    }
                    end = System.nanoTime();
                    testingTime = end - start;
                    testingTime /= Math.pow(10,9);

                    double score = correct/(correct+incorrect)*100 / 100.0;

                    if (oaNames[i].equals("RHC"))
                        rhc_score[ii] += score;
                    if (oaNames[i].equals("SA"))
                        sa_score[ii] += score;
                    if (oaNames[i].equals("GA"))
                        ga_score[ii] += score;
                    
                }
            }
        }

        for (int i = 0; i < 4; i++) {
            rhc_score[i] /= 10;
            ga_score[i] /= 10;
            sa_score[i] /= 10;
        }

        System.out.println("iterations");
        System.out.println("[1, 5, 10, 15]");
        System.out.println(oaNames[0]);
        System.out.println(Arrays.toString(rhc_score));
        System.out.println(oaNames[1]);
        System.out.println(Arrays.toString(sa_score));
        System.out.println(oaNames[2]);
        System.out.println(Arrays.toString(ga_score));
    }

    public static void run_large_iter() {
        double rhc_score[] = new double[5];
        double sa_score[] = new double[5];
        double ga_score[] = new double[5];

        for (int repeats = 0; repeats < 10; repeats++) {
            for(int trainingIterations = 100; trainingIterations <= maxTrainingIterations; trainingIterations += 100) {
                for(int i = 0; i < oa.length; i++) {
                    networks[i] = factory.createClassificationNetwork(
                        new int[] {inputLayer, hiddenLayer, outputLayer});
                    nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
                }
        
                oa[0] = new RandomizedHillClimbing(nnop[0]);
                oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
                oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

                

                for(int i = 0; i < oa.length; i++) {
                    double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                    train(oa[i], networks[i], oaNames[i], trainingIterations); //trainer.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    trainingTime /= Math.pow(10,9);

                    Instance optimalInstance = oa[i].getOptimal();
                    networks[i].setWeights(optimalInstance.getData());

                    double predicted, actual;
                    start = System.nanoTime();
                    for(int j = 0; j < instances.length; j++) {
                        networks[i].setInputValues(instances[j].getData());
                        networks[i].run();

                        predicted = Double.parseDouble(instances[j].getLabel().toString());
                        actual = Double.parseDouble(networks[i].getOutputValues().toString());

                        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                    }
                    end = System.nanoTime();
                    testingTime = end - start;
                    testingTime /= Math.pow(10,9);

                    double score = correct/(correct+incorrect)*100 / 100.0;

                    if (oaNames[i].equals("RHC"))
                        rhc_score[(trainingIterations / 100) - 1] += score;
                    if (oaNames[i].equals("SA"))
                        sa_score[(trainingIterations / 100) - 1] += score;
                    if (oaNames[i].equals("GA"))
                        ga_score[(trainingIterations / 100) - 1] += score;
                }
            }
        }

        for (int i = 0; i < 5; i++) {
            rhc_score[i] /= 10;
            ga_score[i] /= 10;
            sa_score[i] /= 10;
        }

        System.out.println("iterations");
        System.out.println("[100, 200, 300, 400, 500]");
        System.out.println(oaNames[0]);
        System.out.println(Arrays.toString(rhc_score));
        System.out.println(oaNames[1]);
        System.out.println(Arrays.toString(sa_score));
        System.out.println(oaNames[2]);
        System.out.println(Arrays.toString(ga_score));
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int trainingIterations) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[830][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/mammography.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[15]; // 15 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 15; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0] < 1 ? 0 : 1));
        }

        return instances;
    }
}
