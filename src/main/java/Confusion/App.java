package Confusion;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class App {

    private static Instances data;

    public static void main(String[] args) throws Exception {
        loadData("/home/adeiarias/Escritorio/3.MAILA/2.LAUHILABETEA/WEKA/datasets/heart-c.arff");
        confusionMatrix();
    }

    private static void confusionMatrix() throws Exception {
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(data);
        exersice(naiveBayes);
    }

    private static void exersice(Classifier naive) throws Exception {
        //CROSS VALIDATION
        Evaluation evaluation = new Evaluation(data);
        double[] predictions = evaluation.evaluateModel(naive,data);
        createConfusionMatrix(predictions,data,evaluation);
    }

    private static void createConfusionMatrix(double[] prediction, Instances test, Evaluation eval) throws Exception {
        int distinct = test.attribute(test.classIndex()).numValues();
        //ERRENKADAK VALOR REAL
        //ZUTABEAK VALOR ESTIMADO
        int errenkada,zutabea;
        System.out.println("MATRIZEA ESKUZ");
        int[][] matrix = new int[distinct][distinct];

        for(int i = 0; i<matrix.length; i++){
            for(int j = 0; j<matrix[0].length; j++){
                matrix[i][j] = 0;
            }
        }

        for(int i=0; i<prediction.length; i++){
            errenkada = (int)test.get(i).value(test.classIndex());
            zutabea = (int)prediction[i];
            matrix[errenkada][zutabea] += 1;
        }

        for (int x=0; x < matrix.length; x++) {
            System.out.print("|");
            for (int y=0; y < matrix[x].length; y++) {
                System.out.print (matrix[x][y]);
                if (y!=matrix[x].length-1) System.out.print("\t");
            }
            System.out.println("|");
        }

        System.out.println("\n" + eval.toMatrixString());
    }

    private static void loadData(String arffFtixategia) throws Exception {
        DataSource source = new DataSource(arffFtixategia);
        data = source.getDataSet();
        if(data.classIndex() == -1) data.setClassIndex(data.numAttributes()-1);
    }
}
