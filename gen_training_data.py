import argparse
import math
import pickle
import os
import sys
import re
import time
from functools import wraps
import yaml
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from attrdict import AttrDict
import json
import unicodedata

from pydantic import BaseModel, RootModel, field_serializer
from enum import IntEnum
from typing import List, Union

from transformers import TFRobertaModel, AutoTokenizer

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import Row
from pyspark.sql.functions import explode, col, expr, array_join, upper, left, rank, desc, asc, length, arrays_zip
from pyspark.sql.functions import lit, udf, monotonically_increasing_id, pandas_udf, PandasUDFType
from pyspark.sql.functions import unix_timestamp, from_unixtime
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, BinaryType, BooleanType, \
    LongType, DoubleType, FloatType
from openai import OpenAI
from distutils.util import strtobool
from src.utils import prompt_direct_inferring, prompt_direct_inferring_masked, prompt_for_aspect_inferring
from src.preprocess_utils import NLPTextAnalyzer, parse_arguments


#doc = nlp("Barack Obama was born in Hawaii.") # run annotation over a sentence

# polarity_key = {0:positive, 1:negative, 2:neutral}

# indexs 0, 1, 6, 8 from
# laptops_test_gold_pkl_file = '/Users/jordanharris/Code/PycharmProjects/THOR-GEN/data/laptops/Laptops_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'
# indexs 0, 2, from
# laptops_train_gold_pkl_file = '/Users/jordanharris/Code/PycharmProjects/THOR-GEN/data/laptops/Laptops_Train_v2_Implicit_Labeled_preprocess_finetune.pkl'
# data_pkl = {
#         'raw_texts': ["boot time is super fast, around anywhere from 35 seconds to 1 minute.",
#                       "tech support would not fix the problem unless i bought your plan for $150 plus.",
#                       "other than not being a fan of click pads (industry standard these days) and the lousy internal speakers, it's hard for me to find things about this notebook i don't like, especially considering the $350 price tag.",
#                       "no installation disk (dvd) is included.",
#                       "i charge it at night and skip taking the cord with me because of the good battery life.",
#                       "the tech guy then said the service center does not do 1-to-1 exchange and i have to direct my concern to the 'sales' team, which is the retail shop which i bought my netbook from."],
#         'raw_aspect_terms': ["Boot time", "tech support", "price tag", "installation disk (DVD)", "sound output quality", "cord", "service center"],
#         'bert_tokens': [[101, 9573, 2051, 2003, 3565, 3435, 1010, 2105, 5973, 2013, 3486, 3823, 2000, 1015, 3371, 1012, 102],
#                         [101, 6627, 2490, 2052, 2025, 8081, 1996, 3291, 4983, 1045, 4149, 2115, 2933, 2005, 1002, 5018, 4606, 1012, 102],
#                         [101, 2060, 2084, 2025, 2108, 1037, 5470, 1997, 11562, 19586, 1006, 3068, 3115, 2122, 2420, 1007, 1998, 1996, 10223, 6508, 4722, 7492, 1010, 2009, 1005, 1055, 2524, 2005, 2033, 2000, 2424, 2477, 2055, 2023, 14960, 1045, 2123, 1005, 1056, 2066, 1010, 2926, 6195, 1996, 1002, 8698, 3976, 6415, 1012, 102],
#                         [101, 2053, 8272, 9785, 1006, 4966, 1007, 2003, 2443, 1012, 102],
#                         [101, 1045, 3715, 2009, 2012, 2305, 1998, 13558, 2635, 1996, 11601, 2007, 2033, 2138, 1997, 1996, 2204, 6046, 2166, 1012, 102],
#                         [101, 1996, 6627, 3124, 2059, 2056, 1996, 2326, 2415, 2515, 2025, 2079, 1015, 1011, 2000, 1011, 1015, 3863, 1998, 1045, 2031, 2000, 3622, 2026, 5142, 2000, 1996, 1000, 4341, 1000, 2136, 1010, 2029, 2003, 1996, 7027, 4497, 2029, 1045, 4149, 2026, 5658, 8654, 2013, 1012, 102]],
#         'aspect_masks': [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                          [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
#                          [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                          [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
#         'implicits': [False, False, True, True, True, True],
#         'labels': [0, 1, 0, 2, 2, 1]
# }
#
# <sentences>
#     <sentence id="892:1">
#         <text>Boot time is super fast, around anywhere from 35 seconds to 1 minute.</text>
#         <aspectTerms>
#             <aspectTerm term="Boot time" polarity="positive" from="0" to="9" implicit_sentiment="False" opinion_words="fast"/>
#         </aspectTerms>
#     </sentence>
#     <sentence id="1144:1">
#         <text>tech support would not fix the problem unless I bought your plan for $150 plus.</text>
#         <aspectTerms>
#             <aspectTerm term="tech support" polarity="negative" from="0" to="12" implicit_sentiment="False" opinion_words="not fix"/>
#         </aspectTerms>
#     </sentence>
#     <sentence id="2185">
#         <text>High price tag, however.</text>
#         <aspectTerms>
#             <aspectTerm term="price tag" polarity="negative" from="5" to="14" implicit_sentiment="False" opinion_words="High"/>
#         </aspectTerms>
#     </sentence>
#     <sentence id="958:1">
#         <text>Other than not being a fan of click pads (industry standard these days) and the lousy internal speakers, it's hard for me to find things about this notebook I don't like, especially considering the $350 price tag.</text>
#         <aspectTerms>
#             <aspectTerm term="internal speakers" polarity="negative" from="86" to="103" implicit_sentiment="False" opinion_words="lousy"/>
#             <aspectTerm term="price tag" polarity="positive" from="203" to="212" implicit_sentiment="True"/>
#             <aspectTerm term="click pads" polarity="negative" from="30" to="40" implicit_sentiment="True"/>
#         </aspectTerms>
#     </sentence>
#     <sentence id="684:1">
#         <text>excellent in every way.</text>
#     </sentence>
#     <sentence id="282:9">
#         <text>No installation disk (DVD) is included.</text>
#         <aspectTerms>
#             <aspectTerm term="installation disk (DVD)" polarity="neutral" from="3" to="26" implicit_sentiment="True"/>
#         </aspectTerms>
#     </sentence>
#     <sentence id="2185">
#         <text>High price tag, however.</text>
#         <aspectTerms>
#             <aspectTerm term="price tag" polarity="negative" from="5" to="14" implicit_sentiment="False" opinion_words="High"/>
#         </aspectTerms>
#     </sentence>
#     <sentence id="2339">
#         <text>I charge it at night and skip taking the cord with me because of the good battery life.</text>
#         <aspectTerms>
#             <aspectTerm term="cord" polarity="neutral" from="41" to="45" implicit_sentiment="True"/>
#             <aspectTerm term="battery life" polarity="positive" from="74" to="86" implicit_sentiment="False" opinion_words="good"/>
#         </aspectTerms>
#     </sentence>
#     <sentence id="1316">
#         <text>The tech guy then said the service center does not do 1-to-1 exchange and I have to direct my concern to the "sales" team, which is the retail shop which I bought my netbook from.</text>
#         <aspectTerms>
#             <aspectTerm term="service center" polarity="negative" from="27" to="41" implicit_sentiment="True"/>
#             <aspectTerm term="&quot;sales&quot; team" polarity="negative" from="109" to="121" implicit_sentiment="True"/>
#             <aspectTerm term="tech guy" polarity="neutral" from="4" to="12" implicit_sentiment="True"/>
#         </aspectTerms>
#     </sentence>
#

# info
# negation then counter info
# how to rate information density?


def runtime(func):
    @wraps(func)
    def runtime_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return runtime_wrapper


def rest_after_run(sleep_seconds=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Resting for {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)
            print("Starting.")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def json_error_handler(max_retries=3, delay_seconds=8, spec=''):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (json.JSONDecodeError, IndexError, ValueError, AssertionError) as e:
                    print(f"Error: {type(e).__name__} - {e}")
                    print(f"Error decoding {spec} JSON on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {delay_seconds} seconds...")
                        time.sleep(delay_seconds)
                    else:
                        print("Max retries exceeded. Run Canceled")
                        break
            # return none?
            # return None

        return wrapper

    return decorator



class ReasoningStep(BaseModel):
    explanation: str

class AspectTerm(BaseModel):
    aspectTerm: Union[str, List[str]]  # Allow for a single term or a list of terms
    reasoning_steps: List[ReasoningStep]

class AspectResponse(BaseModel):
    aspects: List[AspectTerm]

class Implicitness(BaseModel):
    implicitness: bool
    reasoning_steps: List[ReasoningStep]

class PolarityLabel(IntEnum):
    positive = 0
    negative = 1
    neutral = 2

class Polarity(BaseModel):
    polarity: PolarityLabel
    reasoning_steps: List[ReasoningStep]
    class Config:
        use_enum_values = True  # Serialize Enums to their names

    #@field_serializer('polarity')
    #def serialize_polarity(self, value: PolarityLabel, info):
    #    return value.name  # Serialize to the Enum's name

    def dict(self, *args, **kwargs):
        result = super().dict(*args, **kwargs)
        if not result.get('reasoning_steps'):
            result['reasoning_steps'] = ''  # Convert empty list to empty string
        return result

class ImplicitnessPolarityResponse(BaseModel):
    implicitness: List[Implicitness]
    polarity: List[Polarity]

class genDataset:
    def __init__(self, args, pre_nlp):
        # cwd = os.getcwd()
        #self.pre_nlp = pre_nlp.sort_values(by=['comments'], ascending=True)
        self.pre_nlp = pre_nlp

        config = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        for k, v in vars(args).items():
            setattr(config, k, v)
        self.config = config
        self.config['openai_token'] = os.getenv("OPENAI_API_KEY")
        self.input_file_path = args.raw_file_path
        self.stanza_file_path = args.stanza_file_path
        self.output_file_path = args.out_file_path
        self.raw_text_col = args.raw_text_col
        self.out_text_col = args.out_text_col
        self.batch_size = self.config['gen_batch_size']
        self.output_pkl_path = args.output_pkl_path


        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_path)
        self.spark_session = (SparkSession.builder
                              .master("local[*]")
                              .appName("TiktokComments")
                              .getOrCreate())

        self.csv_schema = StructType([
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
        ])

        #lda_aspect_prob_schema = ArrayType(
        #    StructType([
        #        StructField('topic', FloatType(), True),  # First element of the tuple
        #        StructField('words', StringType(), True)  # Second element of the tuple
        #    ])
        #)

        self.pre_nlp_schema = StructType([
            StructField('comments', StringType(), nullable=False),
            StructField('LDA_aspect_prob', ArrayType(StringType()), True),  # Schema for list of tuples
            StructField('spaCy_tokens', ArrayType(StringType()), nullable=False),
            StructField('POS', ArrayType(StringType()), nullable=False),
            StructField('POS_tags', ArrayType(StringType()), nullable=False),
            StructField('dependencies', ArrayType(StringType()), nullable=False),
            StructField('lemmas', ArrayType(StringType()), nullable=False),
            StructField('heads', ArrayType(StringType()), nullable=False),
            StructField('negations', ArrayType(StringType()), nullable=False),
            StructField('entities', ArrayType(StringType()), nullable=True),
            StructField('labels', ArrayType(StringType()), nullable=True),
            StructField('sentences', ArrayType(StringType()), nullable=True)
        ])

        self.explode_schema = StructType([
            StructField('input_ids', ArrayType(IntegerType()), nullable=False),
            StructField('token_type_ids', ArrayType(IntegerType()), nullable=False),
            StructField('attention_mask', ArrayType(IntegerType()), nullable=False),
            StructField('spaCy_tokens', ArrayType(StringType()), nullable=False),
            StructField('POS', ArrayType(StringType()), nullable=False),
            StructField('POS_tags', ArrayType(StringType()), nullable=False),
            StructField('entities', ArrayType(StringType()), nullable=True),
            StructField('heads', ArrayType(StringType()), nullable=False),
            StructField('labels', ArrayType(StringType()), nullable=True),
            StructField('dependencies', ArrayType(StringType()), nullable=False),
            StructField('negations', ArrayType(StringType()), nullable=True),
            StructField('LDA_aspect_prob', StringType(), nullable=False),
            StructField(self.raw_text_col, StringType(), nullable=True),
        ])

        self.isa_schema = StructType([
            StructField("aspectTerm", StringType(), True),
            StructField("aspect_mask", ArrayType(IntegerType(), True), True),
            StructField("token_ids", ArrayType(IntegerType(), True), True),
            StructField("token_type_ids", ArrayType(IntegerType(), True), True),
            StructField("attention_mask", ArrayType(IntegerType(), True), True),
            StructField("implicitness", BooleanType(), True),
            StructField("polarity", IntegerType(), True),
            StructField("raw_text", StringType(), True),
            StructField("index", LongType(), True)
        ])

        self.final_schema = StructType([
            StructField("Comment", StringType(), True),
            StructField("Comment ID", StringType(), True),
            StructField("Reply to Which Comment", StringType(), True),
            StructField("User ID", StringType(), True),
            StructField("Username", StringType(), True),
            StructField("Nick Name", StringType(), True),
            StructField("Comment Time", StringType(), True),
            StructField("Digg Count", IntegerType(), True),
            StructField("Author Digged", StringType(), True),
            StructField("Reply Count", IntegerType(), True),
            StructField("Pinned to Top", StringType(), True),
            StructField("User Homepage", StringType(), True),
            StructField("shannon_entropy", DoubleType(), True),
            StructField("index", LongType(), True),
            StructField("mutual_information_score", DoubleType(), True),
            StructField("surprisal", DoubleType(), True),
            StructField("perplexity", DoubleType(), True),
            StructField("contextual_mutual_information_score", DoubleType(), True),
            StructField("contextual_surprisal", DoubleType(), True),
            StructField("contextual_perplexity", DoubleType(), True),
            StructField("input_ids", ArrayType(IntegerType(), True), True),
            StructField("token_type_ids", ArrayType(IntegerType(), True), True),
            StructField("attention_mask", ArrayType(IntegerType(), True), True),
            StructField("spaCy_tokens", ArrayType(StringType(), True), True),
            StructField("POS", ArrayType(StringType(), True), True),
            StructField("POS_tags", ArrayType(StringType(), True), True),
            StructField("entities", ArrayType(StringType(), True), True),
            StructField("heads", ArrayType(StringType(), True), True),
            StructField("labels", ArrayType(StringType(), True), True),
            StructField("dependencies", ArrayType(StringType(), True), True),
            StructField("negations", ArrayType(StringType(), True), True),
            StructField("LDA_aspect_prob", StringType(), True),  # Updated to StringType as per the observed schema
            StructField("aspectTerm", StringType(), True),
            StructField("aspect_mask", ArrayType(IntegerType(), True), True),
            StructField("implicitness", BooleanType(), True),
            StructField("polarity", IntegerType(), True),
            StructField("token_ids", ArrayType(IntegerType(), True), True),
            StructField("raw_text", StringType(), True),
            StructField("reasoning", StringType(), True)
        ])

        self.processed_ids = []
        self.remaining_df = None

        self.trigram_probabilities = {}

        self.base_df, self.raw_input_array = self.initialize_df(self.raw_text_col, self.out_text_col)
        self.model = self.config['chat_gpt_model_path']

    @staticmethod
    @udf(returnType=DoubleType())
    def calc_shannon_entropy(text):
        tokens = text.split()
        freq_dist = Counter(tokens)
        total_tokens = len(tokens)
        prob_dist = {token: count / total_tokens for token, count in freq_dist.items()}
        entropy = -sum(prob * math.log2(prob) for prob in prob_dist.values())
        return entropy

    def calc_joint_prob_dist(self, corpus):
        tokens = corpus.lower().split()
        total_tokens = len(tokens)
        freq_dist = Counter(tokens)
        prob_dist = {token: count / total_tokens for token, count in freq_dist.items()}
        bigram_freq_dist = Counter(zip(tokens, tokens[1:]))
        joint_prob_dist = {bigram: count / (total_tokens - 1) for bigram, count in bigram_freq_dist.items()}
        return prob_dist, joint_prob_dist

    def construct_reply_trees(self, comment_pairs):
        reply_tree = {}
        for row in comment_pairs:
            comment_id = row["Comment ID"]
            reply_to_id = row["Reply to Which Comment"]
            if reply_to_id:
                if reply_to_id not in reply_tree:
                    reply_tree[reply_to_id] = []
                reply_tree[reply_to_id].append(comment_id)
        return reply_tree

    def calc_trigram_probabilities(self, corpus):
        tokens = corpus.lower().split()
        total_tokens = len(tokens)

        trigram_freq_dist = Counter(zip(tokens, tokens[1:], tokens[2:]))
        bigram_freq_dist = Counter(zip(tokens, tokens[1:]))

        self.trigram_probabilities = {
            (w1, w2, w3): count / bigram_freq_dist[(w1, w2)]
            for (w1, w2, w3), count in trigram_freq_dist.items()
        }

    def calc_contextual_trigram_probabilities(self, corpus):
        tokens = corpus.lower().split()
        total_tokens = len(tokens)

        trigram_freq_dist = Counter(zip(tokens, tokens[1:], tokens[2:]))
        bigram_freq_dist = Counter(zip(tokens, tokens[1:]))

        trigram_probabilities = {
            (w1, w2, w3): count / bigram_freq_dist[(w1, w2)]
            for (w1, w2, w3), count in trigram_freq_dist.items()
        }
        return trigram_probabilities

    def contextual_surprisal(self, text, trigram_probabilities):
        words = text.lower().split()
        surprisals = []

        for i in range(2, len(words)):
            w1, w2, w3 = words[i - 2], words[i - 1], words[i]
            trigram_prob = trigram_probabilities.get((w1, w2, w3), 1e-10)  # Small probability for unseen trigrams
            surprisal = -math.log2(trigram_prob)
            surprisals.append(surprisal)

        return sum(surprisals) / len(surprisals) if surprisals else 0.0

    def contextual_perplexity(self, text, trigram_probabilities):
        words = text.lower().split()
        N = len(words)
        log_prob_sum = 0.0

        for i in range(2, len(words)):
            w1, w2, w3 = words[i - 2], words[i - 1], words[i]
            trigram_prob = trigram_probabilities.get((w1, w2, w3), 1e-10)  # Small probability for unseen trigrams
            log_prob_sum += math.log2(trigram_prob)

        avg_log_prob = log_prob_sum / (N - 2) if N > 2 else 0
        perplexity = 2 ** (-avg_log_prob)
        return perplexity

    #def calculate_tree_prob_dist(self, tree_comments):
    #     tokens = " ".join(tree_comments).lower().split()
    #     total_tokens = len(tokens)
    #     freq_dist = Counter(tokens)
    #     prob_dist = {token: count / total_tokens for token, count in freq_dist.items()}
    #     bigram_freq_dist = Counter(zip(tokens, tokens[1:]))
    #     joint_prob_dist = {bigram: count / (total_tokens - 1) for bigram, count in bigram_freq_dist.items()}
    #     return prob_dist, joint_prob_dist

    def contextual_MI_Score(self, text, prob_dist, joint_prob_dist):
        tokens = text.lower().split()
        mutual_information_score = 0.0
        for i in range(len(tokens) - 1):
            x, y = tokens[i], tokens[i + 1]
            joint_prob = joint_prob_dist.get((x, y), 1e-10)  # Small probability for unseen bigrams
            marginal_prob_x = prob_dist.get(x, 1e-10)
            marginal_prob_y = prob_dist.get(y, 1e-10)
            mutual_information_score += joint_prob * math.log2(joint_prob / (marginal_prob_x * marginal_prob_y))
        return mutual_information_score

    def construct_contextual_scores(self, df):
        comment_pairs = [row.asDict() for row in df.collect()]
        reply_trees = self.construct_reply_trees(comment_pairs)
        scores = []

        for root_comment, comment_ids in reply_trees.items():
            comment_ids = [root_comment] + comment_ids

            branch = df.filter(df["Comment ID"].isin(comment_ids)).select("Comment ID", "Comment").collect()
            comment_corpus = " ".join([row["Comment"] for row in branch])

            prob_dist, joint_prob_dist = self.calc_joint_prob_dist(comment_corpus)
            contextual_trigram_probabilities = self.calc_contextual_trigram_probabilities(comment_corpus)

            for comment_row in branch:
                comment = comment_row["Comment"]
                mi_score = self.contextual_MI_Score(comment, prob_dist, joint_prob_dist)
                surprisal_score = self.contextual_surprisal(comment, contextual_trigram_probabilities)
                perplexity_score = self.contextual_perplexity(comment, contextual_trigram_probabilities)
                scores.append(
                    (comment_row["Comment ID"], float(mi_score), float(surprisal_score), float(perplexity_score)))

        scores_df = self.spark_session.createDataFrame(scores, ["Comment ID", "contextual_mutual_information_score",
                                                                "contextual_surprisal", "contextual_perplexity"])
        df = df.join(scores_df, on="Comment ID", how="left").orderBy(col("index").desc())
        return df

    def initialize_df(self, raw_text_column, out_text_col):
        base_df = (
            self.spark_session.read
            .schema(self.csv_schema)
            .csv(f"{self.input_file_path}", header=True, inferSchema=True)
            .withColumn("shannon_entropy", self.calc_shannon_entropy(col(raw_text_column)))
            .withColumn("Comment Time", from_unixtime(unix_timestamp(col("Comment Time"), "dd/MM/yyyy, HH:mm:ss")))
            #.orderBy(col("Comment Time"))
            .orderBy([asc('Comment'), desc(length(col('Comment')))]) #asc(col("Comment Time")),
            .withColumn("index", monotonically_increasing_id())
            #.limit(30)
        )
        print('Initialize DF:')
        base_df.show(self.batch_size)

        self.pre_nlp_df = (
            self.spark_session.createDataFrame(self.pre_nlp, self.pre_nlp_schema)
            .orderBy([asc('comments'), desc(length(col('comments')))])
            .withColumn("index", monotonically_increasing_id())  # Adding the 'index' column
        ).orderBy(desc(col("index")))
           # self.processed_df = self.spark_session.createDataFrame([], schema=self.csv_schema)
        #    self.remaining_df = self.spark_session.createDataFrame([], schema=self.csv_schema)

        #------------------------------------------------------------------
        print('Calculating Scores...')

        corpus = base_df.selectExpr("collect_list(Comment) as Comment").collect()[0]["Comment"]
        self.comment_corpus = " ".join(corpus)
        self.calc_trigram_probabilities(self.comment_corpus)
        self.prob_dist, self.joint_prob_dist = self.calc_joint_prob_dist(self.comment_corpus)

        prob_dist_broadcast = self.spark_session.sparkContext.broadcast(self.prob_dist)
        joint_prob_dist_broadcast = self.spark_session.sparkContext.broadcast(self.joint_prob_dist)

        @udf(returnType=DoubleType())
        def mutual_information_udf(text):
            prob_dist = prob_dist_broadcast.value
            joint_prob_dist = joint_prob_dist_broadcast.value
            tokens = text.lower().split()
            mutual_information_score = 0.0
            for i in range(len(tokens) - 1):
                x, y = tokens[i], tokens[i + 1]
                joint_prob = joint_prob_dist.get((x, y), 1e-10)  # Small probability for unseen bigrams
                marginal_prob_x = prob_dist.get(x, 1e-10)
                marginal_prob_y = prob_dist.get(y, 1e-10)
                mutual_information_score += joint_prob * math.log2(joint_prob / (marginal_prob_x * marginal_prob_y))
            return mutual_information_score

        base_df = base_df.withColumn("mutual_information_score", mutual_information_udf(col("Comment")))

        trigram_probabilities_broadcast = self.spark_session.sparkContext.broadcast(self.trigram_probabilities)

        @udf(returnType=DoubleType())
        def surprisal_udf(text):
            trigram_probabilities = trigram_probabilities_broadcast.value
            words = text.lower().split()
            surprisals = []

            for i in range(2, len(words)):
                w1, w2, w3 = words[i - 2], words[i - 1], words[i]
                trigram_prob = trigram_probabilities.get((w1, w2, w3), 1e-10)  # Small probability for unseen trigrams
                surprisal = -math.log2(trigram_prob)
                surprisals.append(surprisal)

            return sum(surprisals) / len(surprisals) if surprisals else 0.0

        base_df = base_df.withColumn("surprisal", surprisal_udf(col("Comment")))

        @udf(returnType=DoubleType())
        def perplexity_udf(text):
            trigram_probabilities = trigram_probabilities_broadcast.value
            words = text.lower().split()
            N = len(words)
            log_prob_sum = 0.0

            for i in range(2, len(words)):
                w1, w2, w3 = words[i - 2], words[i - 1], words[i]
                trigram_prob = trigram_probabilities.get((w1, w2, w3), 1e-10)  # Small probability for unseen trigrams
                log_prob_sum += math.log2(trigram_prob)

            avg_log_prob = log_prob_sum / (N - 2) if N > 2 else 0
            perplexity = 2 ** (-avg_log_prob)
            return perplexity

        base_df = base_df.withColumn("perplexity", perplexity_udf(col("Comment")))

        base_df = self.construct_contextual_scores(base_df)
        base_df = base_df.orderBy(desc(col("index")))
        # base_df = base_df.withColumn("Comment Time", col("Comment Time").cast("timestamp"))
        #base_df = base_df.withColumn("Comment Time",
        #                             from_unixtime(unix_timestamp(col("Comment Time"), "dd/MM/yyyy, HH:mm:ss")))
        print('Information Scores:')
        base_df.show(self.batch_size)
        #------------------------------------------------------------------

        #if os.path.exists(self.output_file_path):
        #    self.processed_df = self.spark_session.read.schema(self.final_schema).parquet(f"{self.output_file_path}")
        #    print('The current parquet df length is: ', self.processed_df.count())
        #    self.processed_df = self.processed_df.orderBy(desc(col("index")))
        #    self.processed_df.show(self.batch_size)
        #    self.processed_ids = self.processed_df.select(out_text_col).distinct().rdd.flatMap(lambda x: x).collect()
        #else:
        #    self.processed_df = self.spark_session.createDataFrame([], schema=self.csv_schema)

        #!!!!!

        #test_base_df = base_df.filter(col("index") == 69)
        #test_base_df.show(truncate=False)

        print('Prenlp length: ',self.pre_nlp_df.count())

        #print('Preprocessed NLP DF')
        #self.pre_nlp_df.show(self.batch_size)
        # !!!!!

        #self.remaining_df = base_df.filter(~base_df[raw_text_column].isin(self.processed_ids))
        #------------------------------------------------------------------
        #self.remaining_df = self.remaining_df.orderBy(asc('comment'), length(col('Comment')).desc())
        #print('Ordered By Alpha Desc')
        #------------------------------------------------------------------
        #print(self.remaining_df.count(), ' Rows remaining')
        #self.remaining_df.show(self.batch_size, truncate=False)




        # RDD stands for Resilient Distributed Dataset, which is a fundamental data structure in Apache Spark.It's a fault-tolerant collection of elements that can be operated on in parallel across a cluster of computers.
        raw_input_array = base_df.select(raw_text_column).rdd.flatMap(lambda x: x).collect()
        if os.path.exists(self.output_file_path):


            self.processed_df = self.spark_session.read.schema(self.final_schema).parquet(f"{self.output_file_path}")
            print('The current parquet df length is: ', self.processed_df.count())
            self.processed_df = self.processed_df.orderBy(desc(col("index")))
            self.processed_df.show(self.batch_size)
            self.processed_ids = self.processed_df.select(out_text_col).distinct().rdd.flatMap(lambda x: x).collect()
            if base_df.count() >= self.processed_df.count():
                self.remaining_df = base_df.filter(~base_df[raw_text_column].isin(self.processed_ids)) # ~ is a negation so is not in
                self.pre_nlp_df = self.pre_nlp_df.filter(~col('comments').isin(self.processed_ids)) # ~ is a negation so is not in
                self.remaining_df = self.remaining_df.orderBy(col("index").desc())
                self.pre_nlp_df = self.pre_nlp_df.orderBy(col("index").desc())
                print(self.remaining_df.count(), ' Rows remaining')
                self.remaining_df.show(self.batch_size, truncate=False)
                self.pre_nlp_df.show(self.batch_size, truncate=False)
                return self.remaining_df, raw_input_array
            else:
                self.remaining_df = self.spark_session.createDataFrame([], schema=self.csv_schema)
                return self.remaining_df, raw_input_array

        else:
            self.remaining_df = base_df.orderBy(col("index").desc())
            return base_df, raw_input_array

    """
    The choice between using BERT or T5 (like flan-t5-base) largely depends on the specific task and the way the model was fine-tuned or trained. Both BERT and T5 are powerful transformer models but are designed with different architectures and objectives:
    1: BERT (Bidirectional Encoder Representations from Transformers) is designed to understand the context of words in a sentence by considering the words that come before and after the target word. It's primarily used for tasks like Named Entity Recognition (NER), sentiment analysis, and question answering.
    2: T5 (Text-to-Text Transfer Transformer) takes a different approach by treating every NLP problem as a text-to-text problem, meaning it converts all NLP tasks into a text-to-text format. This model is versatile and can be used for a variety of tasks, such as translation, summarization, question answering, and more.
    """

    def extract_text_tokens(self, input_array):
        # # RDD stands for Resilient Distributed Dataset, which is a fundamental data structure in Apache Spark.It's a fault-tolerant collection of elements that can be operated on in parallel across a cluster of computers.
        # self.raw_input_array = id_text_token_df.select("raw_texts").rdd.flatMap(lambda x: x).collect()
        # test = self.tokenizer.encode_plus(input_array[0])
        batch_encoded = self.tokenizer.batch_encode_plus(input_array,
                                                         # self.raw_input_array,
                                                         padding=True,
                                                         max_length=self.config.max_length,
                                                         return_tensors=None)
        print(batch_encoded)
        self.bert_tokens = batch_encoded
        return self.bert_tokens

    def convert_lda_aspects(self, lda_aspects):
        # Convert LDA aspects to a JSON string or a formatted string list
        return [json.dumps(aspect) if isinstance(aspect, list) else str(aspect) for aspect in lda_aspects]

    def prep_token_explode(self, batch_df, raw_batch_array):
        #self.pre_nlp_df = self.pre_nlp_df.withColumn('LDA_aspect_prob', self.convert_lda_aspects(col('LDA_aspect_prob')))
        print('Pre NLP DF')
        batch_comments = batch_df.select('Comment').distinct().rdd.flatMap(lambda x: x).collect()

        self.pre_nlp_batch_df = self.pre_nlp_df.filter(col('comments').isin(batch_comments))
        self.pre_nlp_batch_df.show()
        print('Pre NLP Size: ', self.pre_nlp_batch_df.count())
        print('Batch Size: ', batch_df.count())

        input_ids = self.bert_tokens.data['input_ids']
        token_type_ids = self.bert_tokens.data['token_type_ids']
        attention_masks = self.bert_tokens.data['attention_mask']
        spaCy_tokens = self.pre_nlp_batch_df.select('spaCy_tokens').rdd.flatMap(lambda x: x).collect()
        pos = self.pre_nlp_batch_df.select('POS').rdd.flatMap(lambda x: x).collect()
        pos_tags = self.pre_nlp_batch_df.select('POS_tags').rdd.flatMap(lambda x: x).collect()
        entities = self.pre_nlp_batch_df.select('entities').rdd.flatMap(lambda x: x).collect()
        heads = self.pre_nlp_batch_df.select('heads').rdd.flatMap(lambda x: x).collect()
        labels = self.pre_nlp_batch_df.select('labels').rdd.flatMap(lambda x: x).collect()
        dependencies = self.pre_nlp_batch_df.select('dependencies').rdd.flatMap(lambda x: x).collect()
        negations = self.pre_nlp_batch_df.select('negations').rdd.flatMap(lambda x: x).collect()
        lda_aspects = self.pre_nlp_batch_df.select('LDA_aspect_prob').rdd.flatMap(lambda x: x).collect()
        lda_aspects_formatted = self.convert_lda_aspects(lda_aspects)


        zip_data = [
            (
                input_id, token_type_id, attention_mask, spaCy_token, pos_val, pos_tag, entity, head, label, dependency, negation, lda_aspect, raw_input
            )
            for
            input_id, token_type_id, attention_mask, spaCy_token, pos_val, pos_tag, entity, head, label, dependency, negation, lda_aspect, raw_input
            in zip(
                input_ids,
                token_type_ids,
                attention_masks,
                spaCy_tokens,
                pos,
                pos_tags,
                entities,
                heads,
                labels,
                dependencies,
                negations,
                lda_aspects_formatted,
                raw_batch_array
            )
        ]

        schema = StructType([
            StructField('input_ids', ArrayType(IntegerType()), nullable=False),
            StructField('token_type_ids', ArrayType(IntegerType()), nullable=False),
            StructField('attention_mask', ArrayType(IntegerType()), nullable=False),
            StructField('spaCy_tokens', ArrayType(StringType()), nullable=False),
            StructField('POS', ArrayType(StringType()), nullable=False),
            StructField('POS_tags', ArrayType(StringType()), nullable=False),
            StructField('entities', ArrayType(StringType()), nullable=True),
            StructField('heads', ArrayType(StringType()), nullable=False),
            StructField('labels', ArrayType(StringType()), nullable=True),
            StructField('dependencies', ArrayType(StringType()), nullable=False),
            StructField('negations', ArrayType(StringType()), nullable=True),
            StructField('LDA_aspect_prob', StringType(), nullable=False),
            StructField(self.raw_text_col, StringType(), nullable=True),
        ])

        token_nest_df = self.spark_session.createDataFrame(zip_data, schema)

        print('Token Nest DF')
        token_nest_df.show(n=self.batch_size, truncate=False)
        print('Orig Batch')
        batch_df.show(n=self.batch_size, truncate=False)
        batch_df = batch_df.join(token_nest_df, self.raw_text_col, "left").orderBy(desc(col("index")))
        print('Joined Batch')
        batch_df.show()
        return batch_df


    #  PySpark doesn't handle lists of lists automatically without a clear schema.
    @rest_after_run(sleep_seconds=4)
    def explode_df_v2(self, df, uuid, uuid_col_name, nests, exploded_col_name, type):
        if type == dict:
            prep_col = []
            reasoning_col = []
            for x in nests:
                if isinstance(x.aspectTerm, list):
                    prep_col.append(x.aspectTerm)
                    prep = []
                    for y in range(len(x.aspectTerm)):
                        try:
                            prep.append(x.reasoning_steps[y].explanation)
                        except:
                            print()
                    reasoning_col.append(prep)
                else:
                    prep_col.append([x.aspectTerm])
                    reasoning_col.append([x.reasoning_steps[0].explanation])

            zip_data = [(id, nest, reason) for id, nest, reason in zip(uuid, prep_col, reasoning_col)]

            explode_schema = StructType([
                StructField(uuid_col_name, StringType(), True),
                StructField(exploded_col_name, StringType(), True),
                StructField('reasoning', StringType(), True)
            ])

            nests = self.spark_session.createDataFrame(zip_data, [uuid_col_name, exploded_col_name, 'reasoning'], schema=explode_schema)

        unioned_df = df.join(nests, uuid_col_name, "left")
        # Use arrays_zip to combine 'aspectTerm' and 'reasoning'
        unioned_df = unioned_df.withColumn('zipped_col', arrays_zip(exploded_col_name, 'reasoning'))
        print('Joined Batch + Aspects')
        unioned_df.show()
        flat_df = unioned_df.withColumn('zipped_col', explode('zipped_col'))
        flat_df = flat_df.withColumn('aspectTerm', col('zipped_col.aspectTerm'))
        flat_df = flat_df.withColumn('reasoning', col('zipped_col.reasoning'))
        flat_df = flat_df.drop('zipped_col')
        flat_df = flat_df.orderBy(desc(col("index")))
        print('Exploded DF')
        flat_df.show()
        flat_list = flat_df.select(exploded_col_name).rdd.flatMap(lambda x: x).collect()
        assert flat_df.count() == len(flat_list)
        return flat_list, flat_df

    @rest_after_run(sleep_seconds=4)
    def explode_df(self, df, uuid, uuid_col_name, nests, exploded_col_name, type):
        if type == dict:
            prep_col = []
            for x in nests:
                if isinstance(x, dict):
                    if isinstance(x[exploded_col_name], list):
                        prep_col.append(x[exploded_col_name])
                    else:
                        prep_col.append([x[exploded_col_name]])
            zip_data = [(id, nest) for id, nest in zip(uuid, prep_col)]
            nests = self.spark_session.createDataFrame(zip_data, [uuid_col_name, exploded_col_name])
        elif type == list:
            schema = StructType([
                StructField(uuid_col_name, StringType(), False),
                StructField(exploded_col_name, ArrayType(ArrayType(IntegerType())), True)
            ])
            zip_data = [(id, nest) for id, nest in zip(uuid, nests)]
            nests = self.spark_session.createDataFrame(zip_data, schema=schema)

        unioned_df = df.join(nests, uuid_col_name, "left")
        print('Joined Batch + Aspects')
        unioned_df.show()
        flat_df = unioned_df.withColumn(exploded_col_name, explode(unioned_df[exploded_col_name])).orderBy(desc(col("index")))
        print('Exploded DF')
        flat_df.show()
        flat_list = flat_df.select(exploded_col_name).rdd.flatMap(lambda x: x).collect()
        assert flat_df.count() == len(flat_list)
        return flat_list, flat_df


    @json_error_handler(max_retries=3, delay_seconds=2, spec='Base GPT Prompt')
    @rest_after_run(sleep_seconds=2)
    def prompt_gpt(self, role, prompt):
        """
        !!!!!!!THIS IS PAID!!!!!!!
        """
        GPTclient = OpenAI()
        completion = GPTclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ]
        )
        try:
            response = completion.choices[0].message.content
            if response == None:
                print()
            cleaned_response = re.search(r"\[.*\]$", response, re.DOTALL)
            if cleaned_response is None:
                raise ValueError("Could not extract JSON array from the response. Response: " + response)
            cleaned_response = re.sub(r"(?<!\\)'", '"', cleaned_response.string)
            response = json.loads(cleaned_response)
            if response == None:
                print()
        except (json.JSONDecodeError, AssertionError) as e:
            print("Error parsing JSON:", str(e))
            print("Cleaned Response:", cleaned_response)  # Debug the problematic content
            if response == None:
                print()

        print(response)
        assert isinstance(response, list), f"{self.model} output is read to list"
        assert isinstance(response[0], dict), f"{self.model} output is read to list"

        return response

    @json_error_handler(max_retries=5, delay_seconds=2, spec='Base GPT Prompt')
    @rest_after_run(sleep_seconds=2)
    def prompt_gpt_v2(self, role, prompt, response_format):
        """
        !!!!!!!THIS IS PAID!!!!!!!
        """
        GPTclient = OpenAI()
        completion = GPTclient.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
        )
        return completion

    def generate_aspect_mask(self, sentence_tokens, aspect_tokenized):
        mask = [0] * len(sentence_tokens)
        aspect_len = len(aspect_tokenized)
        for i in range(len(sentence_tokens) - aspect_len + 1):
            if sentence_tokens[i:i + aspect_len] == aspect_tokenized:
                for j in range(i, i + aspect_len):
                    mask[j] = 1
        return mask

    def batch_generate_aspect_masks(self, index): #input_ids,
        self.aspect_masks = []
        for i, a in enumerate(self.aspects):
            encoded_aspect_token = self.tokenizer.encode(a, add_special_tokens=False)
            local_index = index[i] % self.batch_size
            self.aspect_masks.append(
                self.generate_aspect_mask(self.bert_tokens.data['input_ids'][local_index], encoded_aspect_token))

        return self.aspect_masks

    def safe_strtobool(self, value):
        if isinstance(value, bool):
            return value
        return bool(strtobool(str(value)))

    def extract_aspects(self, sentence):
        new_context = f'Given the sentence "{sentence}", '
        prompt = new_context + f'which words or phrases are the aspect terms?'
        role = (
            "You are operating as a system that, given a list of sentences, you will identify the core word or phrase in each sentence that are the aspect term or target term of a sentence"
            " that other words in the sentence point to and augment. They do so either implicitly or explicitly. Return the found aspect term(s) in a list where each entry corresponds to a sentence."
            " If no aspect is found, place 'NONE' at the index of that sentence.")
        self.aspects = self.prompt_gpt(role, prompt)
        self.tokenizer.tokenize(self.aspects)
        return self.aspects

    def nlp_batch_for_aspects(self, batch_input, feature_set):
        matching_features = self.pre_nlp[self.pre_nlp['comments'].isin(batch_input)]
        matching_features = matching_features.set_index('comments').reindex(batch_input).reset_index()
        pre_nlp_features = matching_features[feature_set]
        pre_nlp_features_list = pre_nlp_features.to_dict(
            orient='records')
        #pre_nlp_features_str = json.dumps(pre_nlp_features_list, ensure_ascii=False)
        assert len(batch_input) == len(pre_nlp_features_list), "The input arrays must have the same length."
        formatted_prompt = ""
        for index, (sentence, features) in enumerate(zip(batch_input, pre_nlp_features_list)):
            formatted_prompt += f"Input sentence {index}: {sentence}\n"
            formatted_prompt += f"Corresponding NLP features {index}: {features}\n\n"
        return formatted_prompt

    def assert_order_v2(self, aspect_terms, batch_input):
        for aspects, input in zip(aspect_terms, batch_input):
            #aspect_value = aspects['aspectTerm']
            aspect_value = aspects.aspectTerm
            if (aspect_value != 'NONE'):
                if isinstance(aspect_value, list):
                    for a in aspect_value:
                        if a not in input:
                            return False
                else:
                    if aspect_value not in input:
                        return False
        return True

    def assert_order(self, aspect_terms, batch_input):
        for aspects, input in zip(aspect_terms, batch_input):
            aspect_value = aspects['aspectTerm']
            if (aspect_value != 'NONE'):
                if isinstance(aspect_value, list):
                    for a in aspect_value:
                        if a not in input:
                            return False
                else:
                    if aspect_value not in input:
                        return False
        return True

    @json_error_handler(max_retries=3, delay_seconds=2, spec='Aspects')
    @rest_after_run(sleep_seconds=4)
    def batch_extract_aspects(self, nlp_batch, feature_set, max_aspects, batch_input):
        new_context = f'Given these sentences and NLP features "{nlp_batch}", '
        prompt = new_context + f'which words or phrases are the aspect terms?'
        role = (
            f'You are a system that identifies the core word(s) or phrase(s) in a list of sentences, which represent the aspect or target term(s). The max number aspect term to select per sentence is "{max_aspects}"'
            "When considering each sentence also assess all of the preprocessed nlp features at the corresponding index."
            f'The NLP features you will be looking at are the "{feature_set}" if applicable.'
            "LDA (Latent Dirichlet Allocation) aspects are the key topics or themes identified within a document, represented as a distribution of words with associated probabilities, which indicate how relevant each word is to a particular topic."
            "Return the results as a JSON array with proper formatting, where each entry is a JSON object with one key:'aspectTerm'."
            f'IFF the "{max_aspects}" > 1 AND a sentence contains more than one aspect term, list them together as the value for the "aspectTerm"'
            "For example, [{'aspectTerm': 'term0'}, {'aspectTerm': ['term0', 'term1']}, ...].'"
            "If not significant aspect term is found have the value be 'NONE'. Each aspect object corresponds to the input sentence, indexed accordingly."
            "Finally check your output for trailing commas, missing or extra brackets, correct quotation marks, and special characters."
            "Ensure the output contains only this JSON array and no additional text."
        )
        self.aspects = self.prompt_gpt(role, prompt)
        assert len(self.aspects) == self.batch_size
        assert self.assert_order(self.aspects, batch_input), "Aspect terms do not match the input sentences."
        return self.aspects

    @json_error_handler(max_retries=3, delay_seconds=2, spec='AspectResponse')

    def batch_extract_aspects_v2(self, nlp_batch, feature_set, max_aspects, batch_input):
        new_context = f'Given these sentences and NLP features "{nlp_batch}", '
        prompt = new_context + f'which words or phrases are the aspect terms? For each aspect term, provide reasoning steps explaining how it was identified.'

        role = (
            f'You are a system that identifies the core word(s) or phrase(s) in a list of sentences, which represent the aspect or target term(s). '
            f'The max number of aspect terms to select per sentence is "{max_aspects}". '
            "When considering each sentence, also assess all of the preprocessed NLP features at the corresponding index. "
            f'The NLP features you will be looking at are "{feature_set}". '
            "LDA (Latent Dirichlet Allocation) aspects are the key topics or themes identified within a document, represented as a distribution of words with associated probabilities, which indicate how relevant each word is to a particular topic. "
            "Return the results as a JSON array with proper formatting, where each entry corresponds to an input sentence and is a JSON object with the keys 'aspectTerm' and 'reasoning_steps'. "
            "Each 'reasoning_steps' is a list of explanations detailing how the aspect terms were identified. "
            "Remember to process each sentence individually and provide the output in the specified JSON format."
        )

        completion = self.prompt_gpt_v2(role, prompt, AspectResponse)
        self.aspects = completion.choices[0].message.parsed.aspects

        assert len(self.aspects) == self.batch_size
        assert self.assert_order_v2(self.aspects, batch_input), "Aspect terms do not match the input sentences."
        return self.aspects

    def nlp_batch_for_implicitness(self, batch_input, feature_set, aspect_terms):
        assert len(batch_input) == len(feature_set[0]) == len(aspect_terms), "The input arrays must have the same length."

        formatted_prompt = ""
        for index, (sentence, aspects) in enumerate(zip(batch_input, aspect_terms)):

            combined_features = {
                "tokens": feature_set[0][index],
                "POS": feature_set[1][index],
                "POS_tags": feature_set[2][index],
                "heads": feature_set[3][index],
                "dependencies": feature_set[4][index],
                "negations": feature_set[5][index],
            }
            formatted_prompt += f"Input sentence {index}: {sentence}\n"
            formatted_prompt += f"Corresponding NLP features {index}: {combined_features}\n\n"
            formatted_prompt += f"Corresponding Aspect Terms {index}: {aspects}\n\n"
        return formatted_prompt
    def extract_polarity_implicitness(self, aspects):
        new_context = f'Given the sentence "{self.raw_input_array}", and this/these aspect term(s),"{aspects}'
        prompt = new_context + f'determine the polairty (positive, negative or neutral) of aspect term and if it is explicitely or implicitely expressed with respect to the whole sentence?'
        role = (
            "You are operating as a system that, given a sentence & aspect term pair, you will identify the sentiment/polarity of the aspect term within the context of the given sentence."
            "Polarity is either positive, negative or neutral {0, 1, 2}. Then determine if the expression is implicit or explicit (True or False)."
            "Return each found polarity (numeric) and implicitness (boolean) as a tuple (numeric, boolean) in a list where each entry corresponds to the input index of the sentence-aspect pair."
            "If the value of the aspect is NONE, return an object with the polarity calculated as normal but with the 'implicitness' set to 'False'. eg. {'polarity': 2, 'implicitness': 'False'} at the index of that apect input."
        )
        self.polarity_implicitness = self.prompt_gpt(role, prompt)
        return self.polarity_implicitness

    @json_error_handler(max_retries=3, delay_seconds=2, spec='Polarity & Implicits')
    @rest_after_run(sleep_seconds=4)
    def batch_extract_polarity_implicitness(self, nlp_batch, feature_set):
        new_context = f'Given these sentences, NLP features and key aspect terms "{nlp_batch}", with input length: {len(self.aspects)}, '
        # new_context = f'Given this list of lists of sentences, spaCy NLP features and corresponding aspect terms "{zipped_data}" of  length: {self.batch_size}, '
        prompt = new_context + f'determine the polarity (positive, negative or neutral) of aspect term and if it is explicitly or implicitly expressed with respect to the whole sentence?'
        role = (
            "You are operating as a system that, given a list of sentence, spaCy NLP features & aspect terms, you will analyze then identify the sentiment & polarity of the aspect term within the context of the given sentence by filling a json array with that data for later parsing. "
            "Ensure the output contains only this JSON array and no additional leading or trailing text on the formatted json array. "
            "When considering each sentence also assess all of the nlp spaCy features at the corresponding index. "
            f'The NLP features you will be looking at are the "{feature_set}" if applicable. '
            "In dependency parsing, 'heads' refer to the main words (or roots) of phrases that other words depend on, while 'dependencies' describe the grammatical relationships between these dependent words and their heads, such as subjects, objects, and modifiers. "
            "Determine if the expression of the sentiment toward the aspect term is positive neutral or negative and that sentiment expression is implicit or explicit. "
            "# The polarity values to choose from are {0:positive, 1:negative, 2:neutral}. If the expression is implicit, set 'implicitness' to BOOLEAN 'True'; if it is explicit, set 'implicitness' to BOOLEAN 'False'. "
            "Return the results as a JSON array with proper formatting, where each entry is a JSON object with two keys:'polarity' and 'implicitness'. Each entry represents an input sentence-feature-aspect set, indexed accordingly. "
            "If an aspect is 'NONE', return an object with the polarity calculated as normal but with the 'implicitness' set to 'False'. eg. [{'polarity': 1, 'implicitness': 'False'}, {'polarity': 0, 'implicitness': 'True'}, ...] "
            "Be sure to assess every single aspect term and that the length of your output is EXACTLY THE SAME as the length as the INPUT. "
            "Be sure to check for Trailing Commas, Missing/Extra Brackets, Correct Quotation Marks, Special Characters. Do not add the word 'json' before you give the output!"
            #"Ensure the output contains only this JSON array and no additional leading or trailing text on the formatted json array."
        )
        self.polarity_implicitness = self.prompt_gpt(role, prompt)

        try:
            if self.polarity_implicitness is None or self.aspects is None:
                raise TypeError("One or both of the lists are NoneType and cannot be compared.")

            assert len(self.polarity_implicitness) == len(self.aspects), \
                f"Length mismatch: polarity_implicitness ({len(self.polarity_implicitness)}) vs aspects ({len(self.aspects)})"

            implicitness = [self.safe_strtobool(i['implicitness']) for i in self.polarity_implicitness]
        except (json.JSONDecodeError, AssertionError, TypeError, ValueError) as e:
            print("Error occurred:", str(e))
        return self.polarity_implicitness

    @json_error_handler(max_retries=3, delay_seconds=2, spec='Polarity & Implicits')
    @rest_after_run(sleep_seconds=4)
    def batch_extract_polarity_implicitness_v2(self, nlp_batch, feature_set):
        new_context = f'Given these sentences, NLP features and key aspect terms "{nlp_batch}", with input length: {len(self.aspects)}, '
        prompt = new_context + f'determine the polarity (positive, negative or neutral) of aspect term and if it is explicitly or implicitly expressed with respect to the whole sentence?'
        role = (
            "You are operating as a system that, given a list of sentence, spaCy NLP features & aspect terms, you will analyze then identify the sentiment & polarity of the aspect term within the context of the given sentence by filling a json array with that data for later parsing. "
            "Ensure the output contains only this JSON array and no additional leading or trailing text on the formatted json array. "
            "When considering each sentence also assess all of the nlp spaCy features at the corresponding index. "
            f'The NLP features you will be looking at are the "{feature_set}" if applicable. '
            "In dependency parsing, 'heads' refer to the main words (or roots) of phrases that other words depend on, while 'dependencies' describe the grammatical relationships between these dependent words and their heads, such as subjects, objects, and modifiers. "
            "Determine if the expression of the sentiment toward the aspect term is positive neutral or negative and that sentiment expression is implicit or explicit. "
            "Each 'reasoning_steps' is a list of explanations detailing how the implicitness boolean is decided and how the polarity label was assigned. "
        )
        completion = self.prompt_gpt_v2(role, prompt, ImplicitnessPolarityResponse)
        self.implicitness = completion.choices[0].message.parsed.implicitness
        self.polarity = completion.choices[0].message.parsed.polarity


        try:
            assert len(self.polarity) == len(self.aspects), \
                f"Length mismatch: polarity_implicitness ({len(self.polarity)}) vs aspects ({len(self.aspects)})"
            assert len(self.implicitness) == len(self.aspects), \
                f"Length mismatch: polarity_implicitness ({len(self.implicitness)}) vs aspects ({len(self.aspects)})"
        except (json.JSONDecodeError, AssertionError, TypeError, ValueError) as e:
            print("Error occurred:", str(e))
        return self.implicitness, self.polarity

    def transform_df(self, raw_text, token_ids, token_type_ids, attention_masks, aspect_terms, aspect_mask,
                     polarity_implicitness):
        # aspect_terms = [i['aspectTerm'] for i in aspect]
        implicitness = [self.safe_strtobool(i['implicitness']) for i in polarity_implicitness]
        polarity = [i['polarity'] for i in polarity_implicitness]

        rows = [
            Row(
                aspect=aspect_terms[i],
                aspect_mask=aspect_mask[i],
                token_ids=token_ids[i],
                token_type_ids=token_type_ids[i],
                attention_mask=attention_masks[i],
                implicitness=implicitness[i],
                polarity=polarity[i],
                raw_text=raw_text[i],
                index=self.index[i]
            )
            for i in range(len(aspect_terms))
        ]
        final_train_df = self.spark_session.createDataFrame(rows, self.isa_schema)

        final_train_df_columns = final_train_df.columns
        base_df_columns = self.base_df.columns
        batch_df_columns = self.batch_df.columns
#

        print('batch_df')
        self.batch_df.show()
        self.batch_df.cache()
        print('final_train_df')
        final_train_df.show()
        final_train_df.cache()

        #token_ids == input_ids
        full_final_df = self.batch_df.alias('a').join(
            final_train_df.alias('b'),
            (col('a.' + self.raw_text_col) == col('b.' + self.out_text_col)) &
            (col('a.' + 'index') == col('b.' + 'index')) &
            (col('a.' + 'aspectTerm') == col('b.' + 'aspectTerm')),
            "left"
        ).select('a.*', 'b.aspect_mask', 'b.implicitness', 'b.polarity', 'b.token_ids', 'b.raw_text')

        #full_final_df = full_final_df.alias('c').join(
        #    self.base_df.alias('d'),
        #    (col('c.' + self.raw_text_col) == col('d.' + self.raw_text_col)) &
        #    (col('c.' + 'index') == col('d.' + 'index')),
        #    "left"
        #).select('c.*', 'd.mutual_information_score', 'd.surprisal', 'd.perplexity',
        #         'd.contextual_mutual_information_score', 'd.contextual_surprisal', 'd.contextual_perplexity')

        full_final_df = full_final_df.orderBy(col("a.index").desc())
        print('Final batch DF')
        full_final_df.show()
        #full_final_df.printSchema()
        return full_final_df

    def consolidate_reasoning(self, reasoning_list):
        updated_reasonings = []
        for i, (original_reasoning, AspectTerm, comment) in enumerate(reasoning_list):
            try:
                new_reasoning = (original_reasoning + ' ' +
                                 self.implicitness[i].reasoning_steps[0].explanation + ' ' +
                                 self.polarity[i].reasoning_steps[0].explanation)
            except:
                new_reasoning = original_reasoning
            updated_reasonings.append((AspectTerm, comment, new_reasoning))

        updated_reasoning_df = self.spark_session.createDataFrame(
            updated_reasonings, ['aspectTerm', 'Comment', 'reasoning']
        )
        return updated_reasoning_df


    def transform_df_v2(self, raw_text, token_ids, token_type_ids, attention_masks, aspect_terms, aspect_mask,
                     polarity_batch, implicitness_batch):
        # aspect_terms = [i['aspectTerm'] for i in aspect]
        polarity = [i.polarity for i in polarity_batch]
        implicitness = [i.implicitness for i in implicitness_batch]

        rows = [
            Row(
                aspect=aspect_terms[i],
                aspect_mask=aspect_mask[i],
                token_ids=token_ids[i],
                token_type_ids=token_type_ids[i],
                attention_mask=attention_masks[i],
                implicitness=implicitness[i],
                polarity=polarity[i],
                raw_text=raw_text[i],
                index=self.index[i]
            )
            for i in range(len(aspect_terms))
        ]
        final_train_df = self.spark_session.createDataFrame(rows, self.isa_schema)

        final_train_df_columns = final_train_df.columns
        base_df_columns = self.base_df.columns
        batch_df_columns = self.batch_df.columns
        #

        print('batch_df')
        self.batch_df.show()
        self.batch_df.cache()
        print('final_train_df')
        final_train_df.show()
        final_train_df.cache()

        # token_ids == input_ids
        full_final_df = self.batch_df.alias('a').join(
            final_train_df.alias('b'),
            (col('a.' + self.raw_text_col) == col('b.' + self.out_text_col)) &
            (col('a.' + 'index') == col('b.' + 'index')) &
            (col('a.' + 'aspectTerm') == col('b.' + 'aspectTerm')),
            "left"
        ).select('a.*', 'b.aspect_mask', 'b.implicitness', 'b.polarity', 'b.token_ids', 'b.raw_text')
        full_final_df.show()

        full_final_df = full_final_df.orderBy(col("a.index").desc())
        reasoning = full_final_df.select('reasoning', 'aspectTerm', 'Comment').rdd.map(
            lambda row: (row['reasoning'], row['aspectTerm'], row['Comment'])).collect()
        updated_reasoning_df = self.consolidate_reasoning(reasoning)

        full_final_df = full_final_df.alias('original').join(
            updated_reasoning_df.alias('updated'),
            on=['aspectTerm', self.raw_text_col],
            how='left'
        ).select(
            col('original.*'),
            col('updated.reasoning').alias('new_reasoning')
        )

        full_final_df = full_final_df.drop('reasoning').withColumnRenamed('new_reasoning', 'reasoning')

        print('Final batch DF')
        full_final_df.show(truncate=False)
        #full_final_df.printSchema()
        return full_final_df

    @rest_after_run(sleep_seconds=8)
    def write_parquet_file(self, result_df, parquet_path):
        print('Writing df to Parquet file. See data below.')
        result_df.show()
        if not os.path.exists(parquet_path):
            result_df.write.parquet(parquet_path)
        else:
            result_df.write.mode('append').parquet(parquet_path)

    def write_pkl_file(self, pkl_path):
        result_df = self.spark_session.read.schema(self.final_schema).parquet(f"{self.output_file_path}")
        result_df.printSchema()
        try:
            train_df = result_df.select(col('raw_text').alias('raw_texts'),
                                        col('aspectTerm').alias('raw_aspect_terms'),
                                        col('token_ids').alias('bert_tokens'),
                                        col('aspect_mask').alias('aspect_masks'),
                                        col('implicitness').alias('implicits'),
                                        col('polarity').alias('labels')
                                        # col('token_type_ids').alias('token_type_ids'),
                                        # col('attention_mask').alias('attention_mask'),
                                        )

            train_df.show()

            data_rows = train_df.collect()
            # data = [row.asDict() for row in train_df.collect()]
            data_dict = {
                'raw_texts': [row['raw_texts'] for row in data_rows],
                'raw_aspect_terms': [row['raw_aspect_terms'] for row in data_rows],
                'bert_tokens': [row['bert_tokens'] for row in data_rows],
                'aspect_masks': [row['aspect_masks'] for row in data_rows],
                'implicits': [row['implicits'] for row in data_rows],
                'labels': [row['labels'] for row in data_rows]
            }

            with open(pkl_path, 'wb') as file:
                pickle.dump(data_dict, file)
                print("Data successfully written to pickle file.")
        except Exception as e:
            print(f"An error occurred: {e}")

    @runtime
    def run(self):
        # remaining_df = self.input_df
        while self.remaining_df.count() > 0:
            self.batch_df = self.remaining_df.limit(self.batch_size)
            #!!!!SET REMAINING DF TO BE SAVED AS A PARQUET UNTIL ITS DONE THEN DELETE IT SO SCORES DONT HAVE TO CONSTANTLY BE RECALCULATED?
            # ------------------------------------------
            print('Batch DF')
            self.batch_df.show(self.batch_size)
            raw_batch_array = self.batch_df.orderBy(col("index").desc()).select(self.raw_text_col).rdd.flatMap(lambda x: x).collect()
            batch_index = self.batch_df.orderBy(col("index").desc()).select("index").rdd.flatMap(lambda x: x).collect()

            self.extract_text_tokens(raw_batch_array)
            nlp_feature_set = ['spaCy_tokens', 'POS', 'entities', 'labels', 'negations', 'LDA_aspect_prob']
            batch_nlp = self.nlp_batch_for_aspects(raw_batch_array, nlp_feature_set)
            #self.batch_extract_aspects(batch_nlp, nlp_feature_set, 2, raw_batch_array)
            self.batch_extract_aspects_v2(batch_nlp, nlp_feature_set, 2, raw_batch_array)

            self.batch_df = self.prep_token_explode(self.batch_df, raw_batch_array)

            # Bootle Neck
            #self.aspects, self.batch_df = self.explode_df(self.batch_df, raw_batch_array, self.raw_text_col,
            #                                              self.aspects, 'aspectTerm', dict)
            self.aspects, self.batch_df = self.explode_df_v2(self.batch_df, raw_batch_array, self.raw_text_col,
                                                          self.aspects, 'aspectTerm', dict)

            self.batch_df.cache() #To avoid lazy evaluation isses that cause a mismatch you cache to force execution
            print('The exploded batch df is now of size:', self.batch_df.count())
            # / Bootle Neck

            # SpaCy Values
            self.index = self.batch_df.select('index').rdd.flatMap(lambda x: x).collect()
            raw_text = self.batch_df.select(self.raw_text_col).rdd.flatMap(lambda x: x).collect()
            input_ids = self.batch_df.select("input_ids").rdd.flatMap(lambda x: x).collect()
            token_type_ids = self.batch_df.select("token_type_ids").rdd.flatMap(lambda x: x).collect()
            attention_mask = self.batch_df.select("attention_mask").rdd.flatMap(lambda x: x).collect()
            spaCy_tokens = self.batch_df.select("spaCy_tokens").rdd.flatMap(lambda x: x).collect()
            POS = self.batch_df.select("POS").rdd.flatMap(lambda x: x).collect()
            POS_tags = self.batch_df.select("POS_tags").rdd.flatMap(lambda x: x).collect()
            heads = self.batch_df.select("heads").rdd.flatMap(lambda x: x).collect()
            dependencies = self.batch_df.select("dependencies").rdd.flatMap(lambda x: x).collect()
            negations = self.batch_df.select("negations").rdd.flatMap(lambda x: x).collect()

            self.batch_generate_aspect_masks(self.index)
            batch_features_2 = ['spaCy_tokens', 'POS', 'POS_tags', 'heads', 'dependencies', 'negations']
            batch_spaCy_features = [spaCy_tokens, POS, POS_tags, heads, dependencies, negations]
            batch_nlp = self.nlp_batch_for_implicitness(raw_text, batch_spaCy_features, self.aspects)
            #self.batch_extract_polarity_implicitness(batch_nlp, batch_features_2)
            self.implicitness, self.polarity = self.batch_extract_polarity_implicitness_v2(batch_nlp, batch_features_2)
            #self.processed_batch_df = self.transform_df(raw_text, input_ids, token_type_ids, attention_mask,
            #                                            self.aspects, self.aspect_masks, self.polarity_implicitness)

            self.processed_batch_df = self.transform_df_v2(raw_text, input_ids, token_type_ids, attention_mask,
                                                        self.aspects, self.aspect_masks, self.polarity, self.implicitness)
            # ------------------------------------------

            self.write_parquet_file(self.processed_batch_df, self.output_file_path)
            self.processed_ids = self.processed_batch_df.select(self.raw_text_col).rdd.flatMap(lambda x: x).collect()
            self.remaining_df = self.remaining_df.filter(~self.remaining_df[self.raw_text_col].isin(self.processed_ids))
            print('remaining df')

            self.remaining_df.show()
            print('batch finished')
        if not os.path.exists(self.output_pkl_path):
            self.write_pkl_file(self.output_pkl_path)
            print('Run Complete.')
        else:
            print('All data already processed. Terminating.')


if __name__ == '__main__':
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    raw_file_path = './data/raw/TTCommentExporter-7226101187500723498-201-comments.csv'
    #stanza_path = "./data/gen/stanza-7226101187500723498-201.parquet"
    out_parquet_path = "data/gen/train_dataframe.parquet"
    out_pkl_path = './data/gen/Tiktok_Train_Implicit_Labeled_preprocess_finetune.pkl'

    pre_args = parse_arguments(stanza=False, nltk=True, spacy=True)
    preprocessor = NLPTextAnalyzer(args=pre_args)
    comments = preprocessor.read_CSV(raw_file_path)
    nlp_feature_df = preprocessor.construct_nlp_feature_df(comments, 'comments')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='./config/genconfig.yaml', help='config file')
    # parser.add_argument('-i', '--raw_file_path', default='/Users/jordanharris/Code/PycharmProjects/THOR-GEN/data/raw/raw_dev.csv')
    parser.add_argument('-r', '--raw_file_path', default=raw_file_path)
    parser.add_argument('-s', '--stanza_file_path', default='')  #stanza_path)

    parser.add_argument('-r_col', '--raw_text_col', default='Comment')
    parser.add_argument('-o', '--out_file_path', default=out_parquet_path)
    parser.add_argument('-o_col', '--out_text_col', default='raw_text')

    parser.add_argument('-of', '--output_format', default='pkl', choices=['xml', 'json', 'pkl'])
    parser.add_argument('-pkl', '--output_pkl_path', default=out_pkl_path)

    args = parser.parse_args()
    gen = genDataset(args=args, pre_nlp=nlp_feature_df)

    gen.run()
