package com.example;

import org.apache.spark.sql.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class SparkClustring {

    public static void main(String[] args) {
        // Step 1: Initialize Spark
        SparkSession spark = SparkSession.builder()
                .appName("KMeansClusteringExample")
                .master("local[*]")  // Runs locally using all available cores
                .config("spark.sql.legacy.allowUntypedScalaUDF", "true")
                .getOrCreate();
        
        try {
            // Step 2: Load dataset from CSV
            Dataset<Row> dataset = spark.read().format("csv")
                    .option("header", "true")  // Assumes first row has column names
                    .option("inferSchema", "true")  // Auto-detects column types
                    .load("insurance_v1.csv");  // Replace with actual dataset path
            
            // Print the schema to verify column names and types
            System.out.println("Dataset Schema:");
            dataset.printSchema();
            
            // Step 3: Define Numeric and Categorical Columns
            String[] numericCols = {"index","PatientID","age","bmi","bloodpressure","children","claim"};  // Example numeric columns
            String[] categoricalCols = {"gender","diabetic","smoker","region",};      // Example categorical columns
            
            // Step 4: Build a processing pipeline for mixed data types
            List<PipelineStage> pipelineStages = new ArrayList<>();
            
            // Process each categorical column
            for (String categoricalCol : categoricalCols) {
                // Convert string columns to numeric indices
                StringIndexer indexer = new StringIndexer()
                        .setInputCol(categoricalCol)
                        .setOutputCol(categoricalCol + "_index")
                        .setHandleInvalid("keep");  // Handle unknown categories
                
                // One-hot encode the indexed values
                OneHotEncoder encoder = new OneHotEncoder()
                        .setInputCol(categoricalCol + "_index")
                        .setOutputCol(categoricalCol + "_encoded");
                
                pipelineStages.add(indexer);
                pipelineStages.add(encoder);
            }
            
            // Collect all feature columns (numeric + encoded)
            List<String> featureColsList = new ArrayList<>(Arrays.asList(numericCols));
            for (String categoricalCol : categoricalCols) {
                featureColsList.add(categoricalCol + "_encoded");
            }
            String[] featureCols = featureColsList.toArray(new String[0]);
            
            // Create vector of all features
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureCols)
                    .setOutputCol("features")
                    .setHandleInvalid("keep");  // Handle null values
            
            pipelineStages.add(assembler);
            
            // Optional: Standardize features
            StandardScaler scaler = new StandardScaler()
                    .setInputCol("features")
                    .setOutputCol("scaledFeatures")
                    .setWithStd(true)
                    .setWithMean(true);
            
            pipelineStages.add(scaler);
            
            // Create and run the pipeline
            Pipeline pipeline = new Pipeline().setStages(pipelineStages.toArray(new PipelineStage[0]));
            PipelineModel pipelineModel = pipeline.fit(dataset);
            Dataset<Row> featureData = pipelineModel.transform(dataset);
            
            // Display processed features
            System.out.println("Processed Features:");
            featureData.select("features", "scaledFeatures").show(5, false);
            
            // Step 5: Configure and Train K-Means Model
            int numClusters = 3;  // Set the number of clusters (K)
            int numIterations = 20;  // Set number of iterations
            
            KMeans kmeans = new KMeans()
                    .setK(numClusters)
                    .setMaxIter(numIterations)
                    .setFeaturesCol("scaledFeatures")  // Use scaled features
                    .setPredictionCol("prediction")
                    .setSeed(42);  // Set random seed for reproducibility
            
            KMeansModel model = kmeans.fit(featureData);
            
            // Step 6: Output Cluster Centers
            System.out.println("Cluster Centers:");
            Vector[] centers = model.clusterCenters();
            for (int i = 0; i < centers.length; i++) {
                System.out.println("Cluster " + i + ": " + centers[i]);
            }
            
            // Step 7: Make Predictions
            Dataset<Row> predictions = model.transform(featureData);
            
            // Step 8: Display Results
            System.out.println("Clustering Results Sample (with original columns):");
            String[] outputCols = new String[numericCols.length + categoricalCols.length + 1];
            System.arraycopy(numericCols, 0, outputCols, 0, numericCols.length);
            System.arraycopy(categoricalCols, 0, outputCols, numericCols.length, categoricalCols.length);
            outputCols[outputCols.length - 1] = "prediction";
            
            predictions.selectExpr(outputCols).show(10);
            
            // Step 9: Calculate WSSSE (Within Set Sum of Squared Errors)
            // double wssse = model.computeCost(featureData);
            // System.out.println("Within Set Sum of Squared Errors = " + wssse);
            
            // Step 10: Save Results to CSV (optional)
            predictions.selectExpr(outputCols).coalesce(1)
                    .write()
                    .option("header", "true")
                    .csv("kmeans_predictions");
            
            // Save the model (optional)
            model.save("kmeans_model");
            pipelineModel.save("feature_pipeline");
            
            System.out.println("Clustering complete. Results saved to 'kmeans_predictions' directory.");
        } catch (Exception e) {
            System.err.println("Error in K-means clustering:************************************************* " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Stop Spark
            spark.stop();
        }
    }

}
