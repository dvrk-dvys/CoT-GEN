import pickle
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit, udf, monotonically_increasing_id

def read_datafile(path):
    if path.endswith('.pkl'):
        with open(laptops_test_gold_pkl_file, 'rb') as file:
            data = pickle.load(file)
            print(type(data))
            print(data.keys())
            print()
    elif path.endswith('.parquet'):
        spark = SparkSession.builder.master("local[*]").appName("TiktokComments").getOrCreate()

        parquet_df = (
            spark.read
            .parquet(f"{path}", header=True, inferSchema=True)
            .withColumn("index", monotonically_increasing_id())
            # .limit(self.batch_size)
        )
        parquet_df.show(50)
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
    tt_train = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/gen/Tiktok_Train_Implicit_Labeled_preprocess_finetune.pkl'


    # read_datafile(out_parquet_path)
    read_datafile(tt_train)

    print()