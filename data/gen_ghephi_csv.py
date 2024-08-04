import pickle
import shutil

from pyspark.sql import SparkSession, DataFrame, Row, Column
from pyspark.sql.functions import explode, col, expr, array_join, upper, left, rank
from pyspark.sql.functions import lit, udf, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, ArrayType, LongType, DoubleType, TimestampType, BinaryType
import stanza
import subprocess
import gc
import os
import sys
import multiprocessing as mp
import pandas as pd




# # polarity_key = {0:positive, 1:negative, 2:neutral}

class dataViewer:
    def __init__(self):
        self.spark_session = SparkSession.builder.master("local[*]").appName("TiktokComments").getOrCreate()
        self.parquet_df = None

    def close_spark_session(self):
        self.spark_session.stop()
        self.parquet_df = None

    def read_datafile(self, path):
        if path.endswith('.pkl'):
            with open(path, 'rb') as file:
                data = pickle.load(file)
                #print(data)
                # print(type(data))
                # print(data.keys())
            return data

    def load_pkl_to_df(self, path, schema, out=False):
        pandas_df = pd.read_pickle(path)
        pandas_df = pd.DataFrame.from_dict(pandas_df)

        pandas_df['raw_texts'] = pandas_df['raw_texts'].apply(lambda x: [x] if isinstance(x, str) else x)
        pandas_df['raw_aspect_terms'] = pandas_df['raw_aspect_terms'].apply(lambda x: [x] if isinstance(x, str) else x)

        records = pandas_df.to_dict(orient='records')
        rdd = self.spark_session.sparkContext.parallelize(records)
        spark_df = self.spark_session.createDataFrame(rdd, schema=schema)
        spark_df = spark_df.dropDuplicates(['raw_texts'])

        spark_df.show(n=10, truncate=False)
        if out:
            return spark_df
        else:
            self.pkl_df = spark_df

    def debug_datafile(self, path):
        if path.endswith('.parquet'):

            schema = StructType([
                StructField("Comment ID", StringType(), True),
                StructField("Reply to Which Comment", StringType(), True),
                StructField("User ID", StringType(), True),
                StructField("Username", StringType(), True),
                StructField("Nick Name", StringType(), True),
                StructField("Comment", StringType(), True),
                StructField("Comment Time", StringType(), True),
                StructField("Digg Count", IntegerType(), True),
                StructField("Author Digged", StringType(), True),
                StructField("Reply Count", IntegerType(), True),
                StructField("Pinned to Top", StringType(), True),
                StructField("User Homepage", StringType(), True),
                StructField("index", LongType(), True),
                StructField("aspect_mask", ArrayType(IntegerType(), True), True),
                StructField("token_ids", ArrayType(IntegerType(), True), True),
                StructField("token_type_ids", ArrayType(IntegerType(), True), True),
                StructField("attention_mask", ArrayType(IntegerType(), True), True),
                StructField("raw_text", StringType(), True),
                StructField("aspect", StringType(), True),
                StructField("implicitness", BooleanType(), True),  # based on previous error message
                StructField("polarity", IntegerType(), True),
                StructField("shannon_entropy", DoubleType(), True),
                StructField("mutual_information_score", DoubleType(), True),
                StructField("surprisal", DoubleType(), True),
                StructField("perplexity", DoubleType(), True),
                StructField("contextual_mutual_information_score", DoubleType(), True),
                StructField("contextual_surprisal", DoubleType(), True),
                StructField("contextual_perplexity", DoubleType(), True),

            ])

            self.parquet_df = (self.spark_session.read
                          .schema(schema)
                          .parquet(path)
                          # .withColumn("core_index", monotonically_increasing_id())
            )

            df_len = self.parquet_df.count()
            half_len = int(df_len / 2)
            print('The df length is: ', df_len)
            self.parquet_df.show(n=20, truncate=False)
            #text = self.parquet_df.selectExpr("collect_list(Comment) as Comment").collect()[0]["Comment"]

            # parquet_df = (
            #     spark.read
                # .parquet(f"{path}", header=True, inferSchema=True)
                # .withColumn("index", monotonically_increasing_id())
                # .limit(self.batch_size)
            # )
            # parquet_rows = parquet_df.collect()
            # parquet_array = [list(p) for p in parquet_rows]
            # print(parquet_array)
            grooming = self.parquet_df.select("Comment ID", "index", "raw_text", "aspect", "implicitness", "polarity", "shannon_entropy", "mutual_information_score", "surprisal", "perplexity", "contextual_mutual_information_score", "contextual_surprisal", "contextual_perplexity")
            grooming = grooming.orderBy(
                col("perplexity").desc(),
                col("surprisal").desc(),
                col("mutual_information_score").desc(),
                col("contextual_perplexity").desc(),
                col("contextual_surprisal").desc(),
                col("contextual_mutual_information_score").desc(),
            )

    def config_data_vis(self, path):
        schema = StructType([
            StructField("Comment ID", StringType(), True),
            StructField("Reply to Which Comment", StringType(), True),
            StructField("User ID", StringType(), True),
            StructField("Username", StringType(), True),
            StructField("Nick Name", StringType(), True),
            StructField("Comment", StringType(), True),
            StructField("Comment Time", StringType(), True),
            StructField("Digg Count", IntegerType(), True),
            StructField("Author Digged", StringType(), True),
            StructField("Reply Count", IntegerType(), True),
            StructField("Pinned to Top", StringType(), True),
            StructField("User Homepage", StringType(), True),
            StructField("index", LongType(), True),
            StructField("aspect_mask", ArrayType(IntegerType(), True), True),
            StructField("token_ids", ArrayType(IntegerType(), True), True),
            StructField("token_type_ids", ArrayType(IntegerType(), True), True),
            StructField("attention_mask", ArrayType(IntegerType(), True), True),
            StructField("raw_text", StringType(), True),
            StructField("aspect", StringType(), True),
            StructField("implicitness", BooleanType(), True),  # based on previous error message
            StructField("polarity", IntegerType(), True),
            StructField("shannon_entropy", DoubleType(), True),
            StructField("mutual_information_score", DoubleType(), True),
            StructField("surprisal", DoubleType(), True),
            StructField("perplexity", DoubleType(), True),
            StructField("contextual_mutual_information_score", DoubleType(), True),
            StructField("contextual_surprisal", DoubleType(), True),
            StructField("contextual_perplexity", DoubleType(), True),

        ])

        self.parquet_df = (self.spark_session.read
                           .schema(schema)
                           .parquet(path)
                           # .withColumn("core_index", monotonically_increasing_id())
                           )

        df_len = self.parquet_df.count()
        half_len = int(df_len / 2)
        print('The df length is: ', df_len)
        self.parquet_df.show(n=20, truncate=False)

        out_df = self.parquet_df.select("Comment ID", "index", "raw_text", "aspect", "implicitness", "polarity",
                                          "shannon_entropy", "mutual_information_score", "surprisal", "perplexity",
                                          "contextual_mutual_information_score", "contextual_surprisal",
                                          "contextual_perplexity")
        out_df = out_df.orderBy(
            col("perplexity").desc(),
            col("surprisal").desc(),
            col("mutual_information_score").desc(),
            col("contextual_perplexity").desc(),
            col("contextual_surprisal").desc(),
            col("contextual_mutual_information_score").desc(),
        )
        out_df.show(n=20, truncate=False)
        return out_df


    def savetoParquet(self, parquet_path, df):
        df.show(10, truncate=False)

        parquet_df = df.coalesce(1).withColumn("sequential_id", monotonically_increasing_id())
        parquet_df.write.parquet(parquet_path, mode="overwrite")

    def savetoPKL(self, pkl_path, df):
        #pandas_df = df.toPandas(orient='list')
        #for column in pandas_df.columns:
        #    pandas_df[column] = pandas_df[column].apply(lambda x: list(x) if isinstance(x, (list, pd.Series, pd.Index)) else [x])

        if os.path.exists(pkl_path):
            if os.path.isdir(pkl_path):
                shutil.rmtree(pkl_path)
            elif os.path.isfile(pkl_path):
                os.remove(pkl_path)
        with open(pkl_path, 'wb') as f:
            pickle.dump(df, f)
        #df.rdd.saveAsPickleFile(pkl_path)

        #df.to_pickle(pkl_path).collect()

        #data_dict = pandas_df.to_dict(orient='list')
        #with open(pkl_path, 'wb') as file:
        #    pickle.dump(data_dict, file)

    def save_to_csv(self, csv_path, df):
        # Check if the directory exists, if not, create it
        directory = os.path.dirname(csv_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the DataFrame to CSV
        df.write.option("header", "true").csv(csv_path)
        print(f"Data saved to {csv_path}")

    def outputArray(self):
        text = self.parquet_df.selectExpr("collect_list(Comment) as Comment").collect()[0]["Comment"]
        return text


if __name__ == '__main__':
    train_parquet_path = "/Users/joergbln/Desktop/JAH/Code/THOR-GEN/data/gen/train_dataframe.parquet"
    old_parquet_path = "./data/gen/train_dataframe_old.parquet"

    tt_train = '/Users/joergbln/Desktop/JAH/Code/THOR-GEN/data/gen/Tiktok_Train_Implicit_Labeled_preprocess_finetune.pkl'
    gen_csv = "Users/joergbln/Desktop/JAH/Code/THOR-GEN/data/gen/manual_edits/gen_csv/"

    tt_path = "/Users/joergbln/Desktop/JAH/Code/THOR-GEN/data/raw/TTCommentExporter-7226101187500723498-201-comments.csv"
    tt_csv_path = "./gen/7226101187500723498-201-THORGEN.csv"
    parquet_viewer = dataViewer()
    out_df = parquet_viewer.config_data_vis(train_parquet_path)
    parquet_viewer.save_to_csv(tt_csv_path, out_df)

    parquet_viewer.close_spark_session()

    #parquet_viewer.read_datafile(laptops_test_gold_pkl_file)
    #text_col = parquet_viewer.outputArray()
    #parquet_viewer.close_spark_session()


    #nlp_stanza = NLPTextAnalyzer()
    #nlp_stanza.gen_analysis(text_col[0])
    #stanza_process = mp.Process(target=nlp_stanza, args=(text_col[0],))
    #stanza_process.start()
    #stanza_process.join()

    # tt_data = read_datafile(tt_train)
    # laptop_data = read_datafile(laptops_train_gold_pkl_file)
    # {'aspectTerm': 'mistake'}

    print()
    # [('Comment ID', 'string'), ('Reply to Which Comment', 'string'), ('User ID', 'string'), ('Username', 'string'), ('Nick Name', 'string'), ('Comment', 'string'), ('Comment Time', 'timestamp'), ('Digg Count', 'int'), ('Author Digged', 'string'), ('Reply Count', 'int'), ('Pinned to Top', 'string'), ('User Homepage', 'string'), ('index', 'bigint'), ('aspect', 'string'), ('aspect_mask', 'array<bigint>'), ('token_ids', 'array<bigint>'), ('token_type_ids', 'array<bigint>'), ('attention_mask', 'array<bigint>'), ('implicitness', 'boolean'), ('polarity', 'bigint'), ('raw_text', 'string')]
    # [('aspect', 'string'), ('aspect_mask', 'array<bigint>'), ('token_ids', 'array<bigint>'), ('token_type_ids', 'array<bigint>'), ('attention_mask', 'array<bigint>'), ('implicitness', 'boolean'), ('polarity', 'bigint'), ('raw_text', 'string')]
    # [('Comment ID', 'string'), ('Reply to Which Comment', 'string'), ('User ID', 'string'), ('Username', 'string'), ('Nick Name', 'string'), ('Comment', 'string'), ('Comment Time', 'timestamp'), ('Digg Count', 'int'), ('Author Digged', 'string'), ('Reply Count', 'int'), ('Pinned to Top', 'string'), ('User Homepage', 'string'), ('index', 'bigint')]