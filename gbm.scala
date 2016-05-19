// spark-shell --packages com.databricks:spark-csv_2.10:1.4.0

import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy

val data_path = "hdfs://path/file.csv"

val sqlContext = new SQLContext(sc)
val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(data_path)

val char_col_toUse_names = Array("c1","c2","c3","c4", "c5")
val num_col_toUse_names = Array("x1","x2", "x3", "x4")
val target_col_name = "x5"

val distinct_values_in_each_cat_var = for (x <- char_col_toUse_names) yield df.select(x).distinct().count()
######################################################################### With One hot #########################################################################

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
for((x,i) <- char_col_toUse_names.view.zipWithIndex) one_hot_col_names(i) = "one_hot_"+x
val col_name = one_hot_col_names++num_col_toUse_names
val assembler = new VectorAssembler().setInputCols(col_name).setOutputCol("features")


val stages: Array[org.apache.spark.ml.PipelineStage] = index_transformers ++ one_hot_encoders :+ assembler
val pipeline = new Pipeline().setStages(stages)
val indexed_df = pipeline.fit(df).transform(df)

//val ml_df = indexed_df.select(col(target_col_name).cast("int").alias("label"), col("features")).map( row => LabeledPoint( row.getAs[Double]("label"), row.getAs[Any]("features")))

val ml_df = indexed_df.select(col(target_col_name).cast("double").alias("label"), col("features")).map(row => LabeledPoint(row.getDouble(0), row(1).asInstanceOf[Vector]))

val splits = ml_df.randomSplit(Array(0.8, 0.2))
val (trainingData, testData) = (splits(0), splits(1))

val boostingStrategy = BoostingStrategy.defaultParams("Regression")
boostingStrategy.numIterations = 10 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.maxDepth = 3
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

val model = GradientBoostedTrees.train(sc.parallelize(trainingData.collect()), boostingStrategy)

val labelsAndPredictions = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val error = 1.0 * labelsAndPredictions.filter(p => p._1!=0).map(p => (p._2 - p._1).abs/p._1).sum / testData.count()
println(".........................-----------------------=================================================================== Error with one hot encoding: "+ error)

######################################################################### WithOUT One hot #########################################################################


val index_transformers: Array[org.apache.spark.ml.PipelineStage] = char_col_toUse_names.map(
  cname => new StringIndexer()
    .setInputCol(cname)
    .setOutputCol("int_"+cname)
)

var one_hot_col_names = char_col_toUse_names.clone
for((x,i) <- char_col_toUse_names.view.zipWithIndex) one_hot_col_names(i) = "int_"+x
val col_name = one_hot_col_names++num_col_toUse_names
val assembler = new VectorAssembler().setInputCols(col_name).setOutputCol("features")


val stages: Array[org.apache.spark.ml.PipelineStage] = index_transformers :+ assembler
val pipeline = new Pipeline().setStages(stages)
val indexed_df = pipeline.fit(df).transform(df)

//val ml_df = indexed_df.select(col(target_col_name).cast("int").alias("label"), col("features")).map( row => LabeledPoint( row.getAs[Double]("label"), row.getAs[Any]("features")))

val ml_df = indexed_df.select(col(target_col_name).cast("double").alias("label"), col("features")).map(row => LabeledPoint(row.getDouble(0), row(1).asInstanceOf[Vector]))

val splits = ml_df.randomSplit(Array(0.8, 0.2))
val (trainingData, testData) = (splits(0), splits(1))

val boostingStrategy = BoostingStrategy.defaultParams("Regression")
boostingStrategy.numIterations = 10 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.maxDepth = 3
// fill categoricalFeaturesInfo and maxBins from the value of val distinct_values_in_each_cat_var
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]((0,29),(1,3),(2,4),(3,5),(4,107))
boostingStrategy.treeStrategy.maxBins = 107

val model = GradientBoostedTrees.train(sc.parallelize(trainingData.collect()), boostingStrategy)

val labelsAndPredictions = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val error = 1.0 * labelsAndPredictions.filter(p => p._1!=0).map(p => (p._2 - p._1).abs/p._1).sum / testData.count()
println(".........................-----------------------=================================================================== Error with one hot encoding: "+ error)
