package com.flipkart.export;

import com.flipkart.predict.LogisticRegressionPredictor;
import com.flipkart.predict.RandomForestPredictor;
import com.google.common.io.Files;
import junit.framework.TestCase;

import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import org.apache.spark.sql.SQLContext;
import static org.apache.spark.sql.types.DataTypes.*;

public class ExportImportTest extends TestCase {

    private transient JavaSparkContext sc;
    private transient SQLContext sqlContext;
    private transient File tempDir;

    @Before
    public void setUp() {
        sc = new JavaSparkContext("local", "JavaAPISuite");
        sqlContext = new org.apache.spark.sql.SQLContext(sc);
        tempDir = Files.createTempDir();
        tempDir.deleteOnExit();
    }

    @After
    public void tearDown() {
        sc.stop();
        sc = null;
    }

    @Test
    public void testRandomForestBridgeClassification() throws IOException {

        Integer numClasses = 7;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 3;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;

        String datapath = "src/test/resources/classification_test.libsvm";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
        RandomForestModel model = RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo, numTrees,
                featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

        RandomForestExporter randomForestBridgeIn = new RandomForestExporter();
        String modelDump = randomForestBridgeIn.export(model);

        RandomForestPredictor randomForestPredictor = new RandomForestPredictor();
        randomForestPredictor.load(modelDump);

        List<LabeledPoint> testPoints = data.take(10);
        for (LabeledPoint i : testPoints) {
            Vector v = i.features();
            double actual = model.predict(v);
            double predicted = randomForestPredictor.predict(v.toArray());
            assertEquals(actual, predicted);
        }

    }

    @Test
    public void testRandomForestBridgeRegression() throws IOException {

        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        String impurity = "variance";
        int numTrees = 3;
        int maxDepth = 4;
        int maxBins = 32;
        String featureSubsetStrategy = "auto";
        int seed = 12345;

        String datapath = "src/test/resources/regression_test.libsvm";

        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
        RandomForestModel model = RandomForest.trainRegressor(data, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

        RandomForestExporter randomForestBridgeIn = new RandomForestExporter();
        String modelDump = randomForestBridgeIn.export(model);

        RandomForestPredictor randomForestPredictor = new RandomForestPredictor();
        randomForestPredictor.load(modelDump);

        List<LabeledPoint> testPoints = data.take(10);
        for (LabeledPoint i : testPoints) {
            Vector v = i.features();
            double actual = model.predict(v);
            double predicted = randomForestPredictor.predict(v.toArray());
            assertEquals(actual, predicted);
        }

    }

    @Test
    public void testLogisticRegression() throws IOException {

        String datapath = "src/test/resources/binary_classification_test.libsvm";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), datapath).toJavaRDD();
        LogisticRegressionModel lrmodel = new LogisticRegressionWithSGD().run(data.rdd());
        LogisticRegressionExporter logisticRegressionBridgeIn = new LogisticRegressionExporter();

        String modelDump = logisticRegressionBridgeIn.export(lrmodel);

        LogisticRegressionPredictor logisticRegressionPredictor = new LogisticRegressionPredictor();
        logisticRegressionPredictor.load(modelDump);

        lrmodel.clearThreshold();
        List<LabeledPoint> testPoints = data.take(10);
        for (LabeledPoint i : testPoints) {
            Vector v = i.features();
            double actual = lrmodel.predict(v);
            double predicted = logisticRegressionPredictor.predict(v.toArray());
            assertEquals(actual, predicted);
        }

    }

    @Test
    public void testStringIndexer() throws IOException {

        JavaRDD<Row> jrdd = sc.parallelize(Arrays.asList(
                RowFactory.create(0, "a"),
                RowFactory.create(1, "b"),
                RowFactory.create(2, "c"),
                RowFactory.create(3, "a"),
                RowFactory.create(4, "a"),
                RowFactory.create(5, "c")
        ));
        StructType schema = new StructType(new StructField[] {
                createStructField("id", DoubleType, false),
                createStructField("category", StringType, false)
        });

        DataFrame df = sqlContext.createDataFrame(jrdd, schema);

        StringIndexer indexer = new StringIndexer()
                .setInputCol("category")
                .setOutputCol("categoryIndex");

        StringIndexerModel stringIndexerModel = indexer.fit(df);
        DataFrame indexed = stringIndexerModel.transform(df);

        StringIndexerBridge stringIndexerBridge = new StringIndexerBridge();
        System.out.println(stringIndexerBridge.export(stringIndexerModel));

    }
}
