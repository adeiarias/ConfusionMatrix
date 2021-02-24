package Confusion;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.List;

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
        Evaluation evaluation = new Evaluation(data);
        evaluation.evaluateModel(naive,data);
        createConfusionMatrix(evaluation);
    }

    private static void createConfusionMatrix(Evaluation eval) throws Exception {
        int distinct = data.attribute(data.classIndex()).numValues();
        //ROW REAL VALUE
        //COLUMN PREDICTED VALUE
        int row,column;
        System.out.println("CONFUSION MATRIX");
        int[][] matrix = new int[distinct][distinct];

        for(int i = 0; i<matrix.length; i++){
            for(int j = 0; j<matrix[0].length; j++){
                matrix[i][j] = 0;
            }
        }

        List<Prediction> list = eval.predictions();
        for(int i=0; i<list.size(); i++){
            row = (int)list.get(i).actual();
            column = (int)list.get(i).predicted();
            matrix[row][column] += 1;
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
