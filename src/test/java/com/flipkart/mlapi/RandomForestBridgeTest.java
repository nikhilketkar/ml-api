import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;


import scala.Tuple2;
import java.util.HashMap;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;


import org.apache.spark.SparkContext;



public class RandomForestBridgeTest extends TestCase {

    public void testRandomForestBridge()
    {
        SparkConf sparkConf = new SparkConf();
        String master = "local[1]";
        sparkConf.setMaster(master);
        sparkConf.setAppName("Local Spark Unit Test");
        JavaSparkContext sc = new JavaSparkContext(new SparkContext(sparkConf));


        Integer numClasses = 7;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 3; // Use more in practice.
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;
       
        String datapath = "file:///Users/nikhil.ketkar/Desktop/segment.libsvm.txt";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
        RandomForestModel model = RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo, numTrees,
                                                               featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
        assertTrue(true);
        
    }

}
