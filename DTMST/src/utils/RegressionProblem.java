package utils;

import java.io.BufferedReader;

import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Paths;

import weka.core.Instances;


/**
 * @author RFSC
 *class to represent a classification problem
 */
public class RegressionProblem {

        //data structure from weka
        private Instances data;
        //file path of the arff file
        private String filePath;
        //true label
        private Integer trueLabelValue;
        
        
        //constructor
        public RegressionProblem(String filePath) throws IOException{
                this(filePath, new FileReader(filePath));
        }

        //contructor
        public RegressionProblem(String filepath,InputStreamReader ird) throws IOException{

                if(ird != null){
                        //set the filepath of arff file
                        this.setFilePath(filepath);
                        try (BufferedReader br = new BufferedReader(ird))
                        {
                                this.data = new Instances(br);
                                this.data.setClassIndex(data.numAttributes() - 1);

                        } 
                }
        }


        //getters and setters
        public int getNumAttributes(){
                return data.numAttributes();
        }

        public int getNumFeatures(){
                return data.numAttributes();
        }

        public int getNumExamples(){
                return data.numInstances();
        }

        public String getName(){
                return data.relationName();
        }

        public Instances getData() {
                return data;
        }

        public void setData(Instances newData) {
                this.data = newData;
        }

        public String getFilePath() {
                return filePath;
        }

        public void setFilePath(String filePath) {
                this.filePath = Paths.get(filePath).toString();;
        }

        //methods
        @Override
        public String toString() {
                StringBuilder sb = new StringBuilder();
                sb.append("\nname: " + this.getName());
                sb.append("\nfilepath: " + this.getFilePath());
                sb.append("\nNumber Of intances: " + this.getNumExamples());
                sb.append("\nNumber Of Attributes: " + this.getNumAttributes());
                sb.append("\nData:" + this.getData().toString());
                return sb.toString();
        }

        //test this class and the utils methods
        public static void main(String[] args) {



        }

        public Integer getTrueLabelValue() {
                return trueLabelValue;
        }

        public void setTrueLabelValue(Integer trueLabelValue) {
                this.trueLabelValue = trueLabelValue;
        }



}
