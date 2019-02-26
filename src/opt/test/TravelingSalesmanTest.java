package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        double rhc_score[] = new double[5];
        double sa_score[] = new double[5];
        double ga_score[] = new double[5];
        double mm_score[] = new double[5];
        int repeats = 10;
        int iterations[] = {100, 200, 300, 400, 500};

        for (int r = 0; r < repeats; r++) {
            for (int it = 0; it < 5; it++) {
                Random random = new Random();
                // create the random points
                double[][] points = new double[N][2];
                for (int i = 0; i < points.length; i++) {
                    points[i][0] = random.nextDouble();
                    points[i][1] = random.nextDouble();   
                }
                // for rhc, sa, and ga we use a permutation based encoding
                TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
                Distribution odd = new DiscretePermutationDistribution(N);
                NeighborFunction nf = new SwapNeighbor();
                MutationFunction mf = new SwapMutation();
                CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
                HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
                
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations[it]);
                fit.train();
                rhc_score[it] += ef.value(rhc.getOptimal());
                
                SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
                fit = new FixedIterationTrainer(sa, iterations[it]);
                fit.train();
                sa_score[it] += ef.value(sa.getOptimal());
                
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
                fit = new FixedIterationTrainer(ga, iterations[it]);
                fit.train();
                ga_score[it] += ef.value(ga.getOptimal());
                
                // for mimic we use a sort encoding
                ef = new TravelingSalesmanSortEvaluationFunction(points);
                int[] ranges = new int[N];
                Arrays.fill(ranges, N);
                odd = new  DiscreteUniformDistribution(ranges);
                Distribution df = new DiscreteDependencyTree(.1, ranges); 
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
                
                MIMIC mimic = new MIMIC(200, 100, pop);
                fit = new FixedIterationTrainer(mimic, iterations[it]);
                fit.train();
                mm_score[it] += ef.value(mimic.getOptimal());
            }
        }

        for (int i = 0; i < 5; i++) {
            rhc_score[i] /= 10;
            ga_score[i] /= 10;
            sa_score[i] /= 10;
            mm_score[i] /= 10;
        }

        System.out.println("iterations");
        System.out.println("[100, 200, 300, 400, 500]");
        System.out.println("rhc");
        System.out.println(Arrays.toString(rhc_score));
        System.out.println("sa");
        System.out.println(Arrays.toString(sa_score));
        System.out.println("ga");
        System.out.println(Arrays.toString(ga_score));
        System.out.println("mm");
        System.out.println(Arrays.toString(mm_score));
    }
}
