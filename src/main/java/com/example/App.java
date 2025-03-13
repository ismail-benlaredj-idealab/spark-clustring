package com.example;

import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
// $example off$
import org.apache.spark.sql.SparkSession;


public class App 
{
    public static void main( String[] args ){
       // Create a SparkSession.
    SparkSession spark = SparkSession
    .builder()
    .master("local")
    .appName("JavaKMeansExample")
    .getOrCreate();

  // $example on$
  // Loads data.
  Dataset<Row> dataset = spark.read().format("libsvm").load("data/mllib/sample_kmeans_data.txt");

  // Trains a k-means model.
  KMeans kmeans = new KMeans().setK(2).setSeed(1L);
  KMeansModel model = kmeans.fit(dataset);

  // Make predictions
  Dataset<Row> predictions = model.transform(dataset);

  // Evaluate clustering by computing Silhouette score
  ClusteringEvaluator evaluator = new ClusteringEvaluator();

  double silhouette = evaluator.evaluate(predictions);
  System.out.println("Silhouette with squared euclidean distance = " + silhouette);

  // Shows the result.
  Vector[] centers = model.clusterCenters();
  System.out.println("Cluster Centers: ");
  for (Vector center: centers) {
    System.out.println(center);
  }
  // $example off$

  spark.stop();
    }
}
