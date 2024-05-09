import pickle
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit, udf, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, ArrayType, LongType, TimestampType, BinaryType

def read_datafile(path):
    if path.endswith('.pkl'):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            print(data)
            # print(type(data))
            # print(data.keys())
            print()
        return data
    elif path.endswith('.parquet'):
        spark = SparkSession.builder.master("local[*]").appName("TiktokComments").getOrCreate()

        schema = StructType([
            StructField("Comment ID", StringType(), True),
            StructField("Reply to Which Comment", StringType(), True),
            StructField("User ID", StringType(), True),
            StructField("Username", StringType(), True),
            StructField("Nick Name", StringType(), True),
            StructField("Comment", StringType(), True),
            StructField("Comment Time", StringType(), True),  # or TimestampType() if you handle datetime objects
            StructField("Digg Count", IntegerType(), True),
            StructField("Author Digged", StringType(), True),  # assuming 'Yes' or 'No' can be converted to True/False
            StructField("Reply Count", IntegerType(), True),
            StructField("Pinned to Top", StringType(), True),  # assuming 'Yes' or 'No' can be converted to True/False
            StructField("User Homepage", StringType(), True),
            StructField("index", LongType(), True),
            StructField("aspect", StringType(), True),
            StructField("aspect_mask", ArrayType(LongType(), True), True),
            StructField("token_ids", ArrayType(LongType(), True), True),
            StructField("token_type_ids", ArrayType(LongType(), True), True),
            StructField("attention_mask", ArrayType(LongType(), True), True),
            StructField("implicitness", BinaryType(), True),  # based on previous error message
            StructField("polarity", LongType(), True),
            StructField("raw_text", StringType(), True)
        ])

        parquet_df = spark.read.schema(schema).parquet(path)
        print('The df length is: ', parquet_df.count())

        # parquet_df = (
        #     spark.read
            # .parquet(f"{path}", header=True, inferSchema=True)
            # .withColumn("index", monotonically_increasing_id())
            # .limit(self.batch_size)
        # )
        parquet_df.show(15)
        parquet_rows = parquet_df.collect()
        parquet_array = [list(p) for p in parquet_rows]
        print(parquet_array)


if __name__ == '__main__':
    laptops_test_gold_pkl_file = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/laptops/Laptops_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'
    laptops_train_gold_pkl_file = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/laptops/Laptops_Train_v2_Implicit_Labeled_preprocess_finetune.pkl'

    preprocessed_laptops = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/preprocessed/laptops_base_google-flan-t5-base.pkl'
    preprocessed_restauraunts = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/preprocessed/restaurants_base_google-flan-t5-base.pkl'
    old_preprocessed_laptops = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/preprocessed/old_laptops_base_google-flan-t5-base.pkl'

    out_parquet_path = "/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/gen/train_dataframe.parquet"
    old_parquet_path = "/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/gen/train_dataframe_old.parquet"
    tt_train = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/gen/Tiktok_Train_Implicit_Labeled_preprocess_finetune.pkl'


    # old_data = read_datafile(old_parquet_path)
    tt_data = read_datafile(tt_train)
    laptop_data = read_datafile(laptops_train_gold_pkl_file)
    # {'aspectTerm': 'mistake'}

    print()
    # [('Comment ID', 'string'), ('Reply to Which Comment', 'string'), ('User ID', 'string'), ('Username', 'string'), ('Nick Name', 'string'), ('Comment', 'string'), ('Comment Time', 'timestamp'), ('Digg Count', 'int'), ('Author Digged', 'string'), ('Reply Count', 'int'), ('Pinned to Top', 'string'), ('User Homepage', 'string'), ('index', 'bigint'), ('aspect', 'string'), ('aspect_mask', 'array<bigint>'), ('token_ids', 'array<bigint>'), ('token_type_ids', 'array<bigint>'), ('attention_mask', 'array<bigint>'), ('implicitness', 'boolean'), ('polarity', 'bigint'), ('raw_text', 'string')]
    # [('aspect', 'string'), ('aspect_mask', 'array<bigint>'), ('token_ids', 'array<bigint>'), ('token_type_ids', 'array<bigint>'), ('attention_mask', 'array<bigint>'), ('implicitness', 'boolean'), ('polarity', 'bigint'), ('raw_text', 'string')]
    # [('Comment ID', 'string'), ('Reply to Which Comment', 'string'), ('User ID', 'string'), ('Username', 'string'), ('Nick Name', 'string'), ('Comment', 'string'), ('Comment Time', 'timestamp'), ('Digg Count', 'int'), ('Author Digged', 'string'), ('Reply Count', 'int'), ('Pinned to Top', 'string'), ('User Homepage', 'string'), ('index', 'bigint')]