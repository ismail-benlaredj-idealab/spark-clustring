package com.example;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import scala.Tuple2;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class Kmeans {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("JavaKMeansExample").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        
        // Load and parse data
        String path = "insurance_v1.csv";
        String outputPath = "cluster_assignments_2d.csv";
        JavaRDD<String> data = jsc.textFile(path);
        
        // First pass: identify categorical columns and their possible values
        // Assume first line is header
        String header = data.first();
        String[] columns = header.split(",");
        int numColumns = columns.length;
        
        // Skip header for data processing
        JavaRDD<String> dataWithoutHeader = data.filter(line -> !line.equals(header));
        
        // Identify which columns are categorical and collect their unique values
        boolean[] isCategorical = new boolean[numColumns];
        Map<Integer, Set<String>> categoricalValues = new HashMap<>();
        Map<Integer, Map<String, Integer>> categoricalMappings = new HashMap<>();
        
        // First scan: determine which columns are categorical
        List<String[]> rows = dataWithoutHeader.map(line -> line.split(",")).collect();
        for (int i = 0; i < numColumns; i++) {
            boolean categorical = false;
            Set<String> uniqueValues = new HashSet<>();
            
            for (String[] row : rows) {
                String value = row[i].trim();
                uniqueValues.add(value);
                try {
                    Double.parseDouble(value);
                } catch (NumberFormatException e) {
                    categorical = true;
                }
            }
            
            isCategorical[i] = categorical;
            if (categorical) {
                categoricalValues.put(i, uniqueValues);
                
                // Create mapping for one-hot encoding
                Map<String, Integer> valueMap = new HashMap<>();
                int index = 0;
                for (String value : uniqueValues) {
                    valueMap.put(value, index++);
                }
                categoricalMappings.put(i, valueMap);
            }
        }
        
        // Calculate the final vector size after one-hot encoding
        int vectorSize = 0;
        for (int i = 0; i < numColumns; i++) {
            if (isCategorical[i]) {
                vectorSize += categoricalValues.get(i).size();
            } else {
                vectorSize++;
            }
        }
        
        // Second pass: compute min and max for numerical columns for normalization
        double[] minValues = new double[numColumns];
        double[] maxValues = new double[numColumns];
        Arrays.fill(minValues, Double.MAX_VALUE);
        Arrays.fill(maxValues, Double.MIN_VALUE);
        
        for (String[] row : rows) {
            for (int i = 0; i < numColumns; i++) {
                if (!isCategorical[i]) {
                    double value = Double.parseDouble(row[i].trim());
                    minValues[i] = Math.min(minValues[i], value);
                    maxValues[i] = Math.max(maxValues[i], value);
                }
            }
        }
        
        // Find two numerical features for direct plotting
        List<Integer> numericalColumns = new ArrayList<>();
        for (int i = 0; i < numColumns; i++) {
            if (!isCategorical[i]) {
                numericalColumns.add(i);
            }
        }
        
        // For direct plotting, use the first two numerical features if available
        int xAxisColumn = -1;
        int yAxisColumn = -1;
        
        if (numericalColumns.size() >= 2) {
            xAxisColumn = numericalColumns.get(0);
            yAxisColumn = numericalColumns.get(1);
            System.out.println("Selected columns for direct plotting: " + 
                              columns[xAxisColumn] + " and " + columns[yAxisColumn]);
        } else {
            System.out.println("Not enough numerical columns for direct plotting. Will use dimensionality reduction.");
        }
        
        // Convert data to feature vectors with one-hot encoding for categorical variables
        // Store the original data and transformation information for later use
        final boolean[] finalIsCategorical = isCategorical;
        final double[] finalMinValues = minValues;
        final double[] finalMaxValues = maxValues;
        final Map<Integer, Map<String, Integer>> finalCategoricalMappings = categoricalMappings;
        final Map<Integer, Set<String>> finalCategoricalValues = categoricalValues;
        
        // Convert to JavaRDD for processing
        JavaRDD<String[]> rowsRDD = jsc.parallelize(rows);
        
        // Create a mapping of raw data to feature vectors
        final int xx =vectorSize;
        JavaRDD<Vector> parsedData = rowsRDD.map(values -> {
            double[] features = new double[xx];
            
            int featureIndex = 0;
            for (int i = 0; i < values.length; i++) {
                String value = values[i].trim();
                
                if (finalIsCategorical[i]) {
                    // One-hot encoding
                    Map<String, Integer> valueMap = finalCategoricalMappings.get(i);
                    int oneHotIndex = valueMap.get(value);
                    for (int j = 0; j < finalCategoricalValues.get(i).size(); j++) {
                        features[featureIndex++] = (j == oneHotIndex) ? 1.0 : 0.0;
                    }
                } else {
                    // Normalize numerical values
                    double normalizedValue = (Double.parseDouble(value) - finalMinValues[i]) / 
                                             (finalMaxValues[i] - finalMinValues[i]);
                    features[featureIndex++] = normalizedValue;
                }
            }
            
            return Vectors.dense(features);
        });
        
        parsedData.cache();
        
        // Cluster the data using KMeans
        int numClusters = 5;
        int numIterations = 20;
        KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);
        
        // Predict clusters for all data points
        List<Vector> allPoints = parsedData.collect();
        List<Integer> allPredictions = new ArrayList<>();
        
        for (Vector point : allPoints) {
            allPredictions.add(clusters.predict(point));
        }
        
        // Prepare 2D coordinates for plotting
        double[][] plotCoordinates = new double[rows.size()][2];
        
        if (xAxisColumn >= 0 && yAxisColumn >= 0) {
            // Use selected numerical features directly
            for (int i = 0; i < rows.size(); i++) {
                double xValue = Double.parseDouble(rows.get(i)[xAxisColumn].trim());
                double yValue = Double.parseDouble(rows.get(i)[yAxisColumn].trim());
                
                plotCoordinates[i][0] = xValue;
                plotCoordinates[i][1] = yValue;
            }
        } else {
            // Instead of PCA, select two prominent features or use a simplified approach
            // For now, we'll use a simple approach: project to the first two dimensions
            // In a real application, you might want to use a more sophisticated dimensionality reduction
            
            // Simplified approach: find two dimensions with the highest variance
            double[] variances = new double[vectorSize];
            double[] means = new double[vectorSize];
            
            // Calculate means
            for (Vector point : allPoints) {
                for (int i = 0; i < vectorSize; i++) {
                    means[i] += point.apply(i);
                }
            }
            for (int i = 0; i < vectorSize; i++) {
                means[i] /= allPoints.size();
            }
            
            // Calculate variances
            for (Vector point : allPoints) {
                for (int i = 0; i < vectorSize; i++) {
                    variances[i] += Math.pow(point.apply(i) - means[i], 2);
                }
            }
            for (int i = 0; i < vectorSize; i++) {
                variances[i] /= allPoints.size();
            }
            
            // Find two dimensions with highest variance
            int dim1 = 0;
            int dim2 = 1;
            for (int i = 2; i < vectorSize; i++) {
                if (variances[i] > variances[dim1]) {
                    dim2 = dim1;
                    dim1 = i;
                } else if (variances[i] > variances[dim2]) {
                    dim2 = i;
                }
            }
            
            // Project to these two dimensions
            for (int i = 0; i < allPoints.size(); i++) {
                plotCoordinates[i][0] = allPoints.get(i).apply(dim1);
                plotCoordinates[i][1] = allPoints.get(i).apply(dim2);
            }
            
            System.out.println("Using dimensions with highest variance for 2D projection");
        }
        
        // Write the results to a CSV file suitable for plotting
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
            // Write header
            if (xAxisColumn >= 0 && yAxisColumn >= 0) {
                writer.write(columns[xAxisColumn] + "," + columns[yAxisColumn] + ",cluster\n");
            } else {
                writer.write("Dimension1,Dimension2,cluster\n");
            }
            
            // Write data rows with cluster assignments and 2D coordinates for plotting
            for (int i = 0; i < rows.size(); i++) {
                StringBuilder sb = new StringBuilder();
                
                // Write 2D coordinates
                sb.append(plotCoordinates[i][0]).append(",");
                sb.append(plotCoordinates[i][1]).append(",");
                
                // Add cluster assignment
                sb.append(allPredictions.get(i));
                sb.append("\n");
                
                writer.write(sb.toString());
            }
            
            System.out.println("Cluster assignments with 2D coordinates saved to " + outputPath);
        } catch (IOException e) {
            System.err.println("Error writing to CSV file: " + e.getMessage());
            e.printStackTrace();
        }
        
        // Write a second file with complete data for further analysis
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("complete_cluster_assignments.csv"))) {
            // Write header with additional columns for plotting coordinates
            if (xAxisColumn >= 0 && yAxisColumn >= 0) {
                writer.write(header + ",plot_x,plot_y,cluster\n");
            } else {
                writer.write(header + ",dim1,dim2,cluster\n");
            }
            
            // Write data rows with cluster assignments
            for (int i = 0; i < rows.size(); i++) {
                StringBuilder sb = new StringBuilder();
                
                // Write original data values
                for (int j = 0; j < rows.get(i).length; j++) {
                    sb.append(rows.get(i)[j]);
                    sb.append(",");
                }
                
                // Add plotting coordinates
                sb.append(plotCoordinates[i][0]).append(",");
                sb.append(plotCoordinates[i][1]).append(",");
                
                // Add cluster assignment
                sb.append(allPredictions.get(i));
                sb.append("\n");
                
                writer.write(sb.toString());
            }
            
            System.out.println("Complete data with cluster assignments saved to complete_cluster_assignments.csv");
        } catch (IOException e) {
            System.err.println("Error writing to CSV file: " + e.getMessage());
            e.printStackTrace();
        }
        
        // Output to console
        System.out.println("\nCluster centers:");
        for (Vector center : clusters.clusterCenters()) {
            System.out.println(" " + center);
        }
        
        double cost = clusters.computeCost(parsedData.rdd());
        System.out.println("Within Set Sum of Squared Errors = " + cost);
        
        // Additional output for visualization assistance
        System.out.println("\nVisualization files created:");
        System.out.println("1. " + outputPath + " - Contains just the 2D coordinates and cluster assignments");
        System.out.println("2. complete_cluster_assignments.csv - Contains all original data plus 2D coordinates and cluster assignments");
        System.out.println("\nTo plot this data:");
        System.out.println("1. Use a scatter plot with the first column as X, second as Y, and color points by the cluster column");
        System.out.println("2. Sample plotting code in Python:");
        System.out.println("   import pandas as pd");
        System.out.println("   import matplotlib.pyplot as plt");
        System.out.println("   df = pd.read_csv('" + outputPath + "')");
        System.out.println("   plt.figure(figsize=(10, 8))");
        System.out.println("   for cluster in df['cluster'].unique():");
        System.out.println("       subset = df[df['cluster'] == cluster]");
        System.out.println("       plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1], label=f'Cluster {cluster}')");
        System.out.println("   plt.legend()");
        System.out.println("   plt.title('K-means Clustering Results')");
        System.out.println("   plt.xlabel(df.columns[0])");
        System.out.println("   plt.ylabel(df.columns[1])");
        System.out.println("   plt.show()");
        
        jsc.stop();
    }
}