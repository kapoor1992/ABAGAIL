package opt.test;

import java.util.Arrays;
import java.util.Random;
import opt.ga.NQueensFitnessFunction;
import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * @author kmanda1
 * @version 1.0
 */
public class NQueensTest {
    /** The n value */
    private static final int N = 10;
    /** The t value */
    
    public static void main(String[] args) {
        double rhc_score[] = new double[5];
        double sa_score[] = new double[5];
        double ga_score[] = new double[5];
        double mm_score[] = new double[5];
        int repeats = 10;
        int iterations[] = {100, 200, 300, 400, 500};

        for (int r = 0; r < repeats; r++) {
            for (int it = 0; it < 5; it++) {
                int[] ranges = new int[N];
                Random random = new Random(N);
                for (int i = 0; i < N; i++) {
                    ranges[i] = random.nextInt();
                }
                NQueensFitnessFunction ef = new NQueensFitnessFunction();
                Distribution odd = new DiscretePermutationDistribution(N);
                NeighborFunction nf = new SwapNeighbor();
                MutationFunction mf = new SwapMutation();
                CrossoverFunction cf = new SingleCrossOver();
                Distribution df = new DiscreteDependencyTree(.1); 
                HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
                
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations[it]);
                fit.train();
                long starttime = System.currentTimeMillis();
                rhc_score[it] += ef.value(rhc.getOptimal());
                
                SimulatedAnnealing sa = new SimulatedAnnealing(1E1, .1, hcp);
                fit = new FixedIterationTrainer(sa, iterations[it]);
                fit.train();
                
                starttime = System.currentTimeMillis();
                sa_score[it] += ef.value(sa.getOptimal());
                
                starttime = System.currentTimeMillis();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 0, 10, gap);
                fit = new FixedIterationTrainer(ga, iterations[it]);
                fit.train();
                ga_score[it] += ef.value(ga.getOptimal());
                
                starttime = System.currentTimeMillis();
                MIMIC mimic = new MIMIC(200, 10, pop);
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
