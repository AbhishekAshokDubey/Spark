import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder,VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.regression.GBTRegressionModel
import org.apache.spark.sql.types._

import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, CrossValidator}
import org.apache.spark.ml.evaluation.RegressionEvaluator


val data_path_test = "path/TEST_DATA.csv"
val data_path_train = "path/TRAIN_DATA.csv"


val char_col_toUse_names = Array("AREA","ASSET_NAME","CUSTOMER_NAME", "GEO_MARKET", "HOLE_SIZE_RANGE","LOCATION_CODE","RIG_NAME","RIG_TYPE","SUB_SEGMENT", "WELL_ENVIRONMENT","WELL_GEOMETRY","WELL_TYPE")
val num_col_toUse_names = Array("HS_DURATION")
val target_col_name = "ASSET_DAYS"
val do_char_to_OneHot = false;

val all_toUse_name = char_col_toUse_names++num_col_toUse_names :+target_col_name

//val sqlContext = new SQLContext(sc)
val df_test = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(data_path_test).select(all_toUse_name.head, all_toUse_name.tail: _*).withColumn(target_col_name, col(target_col_name) cast DoubleType)
val df_train = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(data_path_train).select(all_toUse_name.head, all_toUse_name.tail: _*).withColumn(target_col_name, col(target_col_name) cast DoubleType)


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
if(do_char_to_OneHot){for((x,i) <- char_col_toUse_names.view.zipWithIndex) one_hot_col_names(i) = "one_hot_"+x} else{for((x,i) <- char_col_toUse_names.view.zipWithIndex) one_hot_col_names(i) = "int_"+x}

val col_name = one_hot_col_names++num_col_toUse_names
val assembler = new VectorAssembler().setInputCols(col_name).setOutputCol("features")

val gbt_model = new GBTRegressor().setLabelCol(target_col_name).setFeaturesCol("features").setMaxIter(200).setMaxDepth(5).setMaxBins(1900)

val gbt_model = new GBTRegressor().setLabelCol(target_col_name).setFeaturesCol("features").setMaxBins(1900)

var stages: Array[org.apache.spark.ml.PipelineStage] = index_transformers :+ assembler :+ gbt_model
if(do_char_to_OneHot){stages = index_transformers ++ one_hot_encoders :+ assembler :+ gbt_model}

val pipeline = new Pipeline().setStages(stages)

val paramGrid = new ParamGridBuilder().addGrid(gbt_model.maxIter, Array(100, 200)).addGrid(gbt_model.maxDepth, Array(2, 5, 10)).build()

val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)
val cvModel = cv.fit(df_train)
val df_test_with_predictions = cvModel.transform(df_test)


val trainValidationSplit = new TrainValidationSplit().setEstimator(pipeline).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)
val trainValidationSplitModel = trainValidationSplit.fit(df_train)
