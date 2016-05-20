// spark-shell --packages com.databricks:spark-csv_2.10:1.4.0

import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.BoostingStrategy

val data_path = "hdfs://path/file.csv"

val sqlContext = new SQLContext(sc)
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(data_path)

val char_col_toUse_names = Array("c1","c2","c3","c4", "c5")
val num_col_toUse_names = Array("x1","x2", "x3", "x4")
val target_col_name = "x5"
val do_char_to_OneHot = false;

val distinct_values_in_each_cat_var = for (x <- char_col_toUse_names) yield df.select(x).distinct().count()

val index_transformers: Array[org.apache.spark.ml.PipelineStage] = char_col_toUse_names.map(
  cname => new StringIndexer()
    .setInputCol(cname)
    .setOutputCol("int_"+cname)
)

val one_hot_encoders: Array[org.apache.spark.ml.PipelineStage] = char_col_toUse_names.map(
  cname => new OneHotEncoder()
    .setInputCol("int_"+cname)
    .setOutputCol("one_hot_"+cname)
)

var one_hot_col_names = char_col_toUse_names.clone
if(do_char_to_OneHot){for((x,i) <- char_col_toUse_names.view.zipWithIndex) one_hot_col_names(i) = "one_hot_"+x}

val col_name = one_hot_col_names++num_col_toUse_names
val assembler = new VectorAssembler().setInputCols(col_name).setOutputCol("features")

var stages: Array[org.apache.spark.ml.PipelineStage] = index_transformers :+ assembler
if(do_char_to_OneHot){stages = index_transformers ++ one_hot_encoders :+ assembler}

val pipeline = new Pipeline().setStages(stages)
val indexed_df = pipeline.fit(df).transform(df)

//val ml_df = indexed_df.select(col(target_col_name).cast("int").alias("label"), col("features")).map( row => LabeledPoint( row.getAs[Double]("label"), row.getAs[Any]("features")))

val ml_df = indexed_df.select(col(target_col_name).cast("double").alias("label"), col("features")).map(row => LabeledPoint(row.getDouble(0), row(1).asInstanceOf[Vector]))

val splits = ml_df.randomSplit(Array(0.8, 0.2))
val (trainingData, testData) = (splits(0), splits(1))

val numTrees = 3 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "variance"
val maxDepth = 4
val maxBins = 107
var categoricalFeaturesInfo = Map[Int, Int]()
if(!do_char_to_OneHot){categoricalFeaturesInfo = Map[Int, Int]((0,29),(1,3),(2,4),(3,5),(4,107))}

val model = RandomForest.trainRegressor(sc.parallelize(trainingData.collect()), categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
val labelsAndPredictions = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val error = 1.0 * labelsAndPredictions.filter(p => p._1!=0).map(p => (p._2 - p._1).abs/p._1).sum / testData.count()
println(".........................-----------------------=================================================================== Error with one hot encoding: "+ error)
