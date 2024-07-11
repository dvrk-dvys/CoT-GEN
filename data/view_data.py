import pickle
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import explode, col, expr, array_join, upper, left, rank
from pyspark.sql.functions import lit, udf, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, ArrayType, LongType, DoubleType, TimestampType, BinaryType
import stanza
import subprocess
import gc
import os
import sys
import multiprocessing as mp




# # polarity_key = {0:positive, 1:negative, 2:neutral}

def restart_kernel():
    os.execv(sys.executable, ['python'] + sys.argv)


class NLPTextAnalyzer:
    def __init__(self):
        stanza.download('en')
        self.nlp = stanza.Pipeline('en')
    def gen_analysis(self, text):
        doc = self.nlp(text)
        print(doc)

class dataViewer:
    def __init__(self):
        self.spark_session = SparkSession.builder.master("local[*]").appName("TiktokComments").getOrCreate()
        self.parquet_df = None

    def close_spark_session(self):
        self.spark_session.stop()
        self.parquet_df = None
        gc.collect()

    def read_datafile(self, path):
        if path.endswith('.pkl'):
            with open(path, 'rb') as file:
                data = pickle.load(file)
                print(data)
                # print(type(data))
                # print(data.keys())
                print()
            return data
        elif path.endswith('.parquet'):

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
                col("contextual_mutual_information_score").desc()
                #col("perplexity").asc(),
                #col("surprisal").asc(),
                #col("mutual_information_score").asc(),
                #col("contextual_perplexity").asc(),
                #col("contextual_surprisal").asc(),
                #col("contextual_mutual_information_score").asc()
            )
    def savetoCSV(self, csv_path, df):
        df.show(10, truncate=False)

        csv_df = df.coalesce(1).withColumn("sequential_id", monotonically_increasing_id())
        csv_df.write.csv(csv_path, header=True, mode="overwrite")

    def outputArray(self):
        text = self.parquet_df.selectExpr("collect_list(Comment) as Comment").collect()[0]["Comment"]
        return text


if __name__ == '__main__':
    os.environ['JAVA_OPTS'] = '-Xmx4G'  # Set maximum Java heap size
    os.environ['SPARK_DRIVER_MEMORY'] = '4g'
    os.environ['SPARK_EXECUTOR_MEMORY'] = '4g'

    laptops_test_gold_pkl_file = '/data/laptops/Laptops_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'
    laptops_train_gold_pkl_file = '/data/laptops/Laptops_Train_v2_Implicit_Labeled_preprocess_finetune.pkl'

    preprocessed_laptops = '/Users/jordanharris/Code/PycharmProjects/THOR-GEN/data/preprocessed/laptops_base_google-flan-t5-base.pkl'
    preprocessed_restauraunts = '/Users/jordanharris/Code/PycharmProjects/THOR-GEN/data/preprocessed/restaurants_base_google-flan-t5-base.pkl'
    old_preprocessed_laptops = '/Users/jordanharris/Code/PycharmProjects/THOR-GEN/data/preprocessed/old_laptops_base_google-flan-t5-base.pkl'


    train_parquet_path = "/Users/joergbln/Desktop/JAH/Code/THOR-GEN/data/gen/train_dataframe.parquet"
    old_parquet_path = "./data/gen/train_dataframe_old.parquet"

    tt_train = '/Users/jordanharris/Code/PycharmProjects/THOR-GEN/data/gen/Tiktok_Train_Implicit_Labeled_preprocess_finetune.pkl'

    gen_csv = "/Users/jordanharris/Code/PycharmProjects/THOR-GEN/data/gen/manual_edits/gen_csv/"
    parquet_viewer = dataViewer()
    parquet_viewer.read_datafile(train_parquet_path)
    text_col = parquet_viewer.outputArray()
    parquet_viewer.close_spark_session()


    manager = mp.Manager()
    return_dict = manager.dict()


    nlp_stanza = NLPTextAnalyzer()
    #nlp_stanza.gen_analysis(text_col[0])
    stanza_process = mp.Process(target=nlp_stanza, args=(text_col[0],))
    stanza_process.start()
    stanza_process.join()

    # tt_data = read_datafile(tt_train)
    # laptop_data = read_datafile(laptops_train_gold_pkl_file)
    # {'aspectTerm': 'mistake'}

    print()
    # [('Comment ID', 'string'), ('Reply to Which Comment', 'string'), ('User ID', 'string'), ('Username', 'string'), ('Nick Name', 'string'), ('Comment', 'string'), ('Comment Time', 'timestamp'), ('Digg Count', 'int'), ('Author Digged', 'string'), ('Reply Count', 'int'), ('Pinned to Top', 'string'), ('User Homepage', 'string'), ('index', 'bigint'), ('aspect', 'string'), ('aspect_mask', 'array<bigint>'), ('token_ids', 'array<bigint>'), ('token_type_ids', 'array<bigint>'), ('attention_mask', 'array<bigint>'), ('implicitness', 'boolean'), ('polarity', 'bigint'), ('raw_text', 'string')]
    # [('aspect', 'string'), ('aspect_mask', 'array<bigint>'), ('token_ids', 'array<bigint>'), ('token_type_ids', 'array<bigint>'), ('attention_mask', 'array<bigint>'), ('implicitness', 'boolean'), ('polarity', 'bigint'), ('raw_text', 'string')]
    # [('Comment ID', 'string'), ('Reply to Which Comment', 'string'), ('User ID', 'string'), ('Username', 'string'), ('Nick Name', 'string'), ('Comment', 'string'), ('Comment Time', 'timestamp'), ('Digg Count', 'int'), ('Author Digged', 'string'), ('Reply Count', 'int'), ('Pinned to Top', 'string'), ('User Homepage', 'string'), ('index', 'bigint')]