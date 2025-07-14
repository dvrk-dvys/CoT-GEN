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
from addict import Dict

import json
import unicodedata

from pydantic import BaseModel, RootModel, field_serializer
from enum import IntEnum
from typing import List, Union

from transformers import TFRobertaModel, AutoTokenizer, AutoModelForCausalLM
import torch

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import Row
from pyspark.sql.functions import explode, col, expr, array_join, upper, rank, desc, asc, length, arrays_zip
from pyspark.sql.functions import lit, udf, monotonically_increasing_id, pandas_udf
from pyspark.sql.functions import unix_timestamp, from_unixtime, split, sum as spark_sum, log2, collect_list
from pyspark.sql.functions import when, isnan, isnull, size, array, struct, map_values, map_keys
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, BinaryType, BooleanType, \
    LongType, DoubleType, FloatType, MapType

from openai import OpenAI
from distutils.util import strtobool
from src.utils import prompt_direct_inferring, prompt_direct_inferring_masked, prompt_for_aspect_inferring
from src.preprocess_utils import NLPTextAnalyzer, parse_arguments


def _calculate_tree_stats_standalone(row):
    """Standalone function to calculate statistics for each comment tree."""
    reply_to_id = row["Reply to Which Comment"]
    tree_text = row["tree_comment_text"]
    
    if not tree_text:
        # Return empty stats for null trees
        return (reply_to_id, "{}", "{}", "{}")
    
    # Calculate distributions for this tree using local functions
    prob_dist, joint_prob_dist = calc_joint_prob_dist_native_local(tree_text)
    trigram_probabilities = calc_trigram_probabilities_native_local(tree_text)
    
    # Serialize to JSON strings for DataFrame storage
    return (
        reply_to_id,
        json.dumps(prob_dist),
        json.dumps(joint_prob_dist),
        json.dumps(trigram_probabilities)
    )


def calc_joint_prob_dist_native_local(corpus_text):
    """Local version of joint probability calculation for tree statistics."""
    tokens = corpus_text.lower().split()
    total_tokens = len(tokens)
    
    if total_tokens == 0:
        return {}, {}
    
    # Token probabilities
    freq_dist = Counter(tokens)
    prob_dist = {token: count / total_tokens for token, count in freq_dist.items()}
    
    # Bigram joint probabilities
    if total_tokens < 2:
        return prob_dist, {}
    
    bigrams = [f"{tokens[i]},{tokens[i+1]}" for i in range(total_tokens-1)]
    bigram_freq_dist = Counter(bigrams)
    joint_prob_dist = {bigram: count / (total_tokens - 1) 
                      for bigram, count in bigram_freq_dist.items()}
    
    return prob_dist, joint_prob_dist


def calc_trigram_probabilities_native_local(corpus_text):
    """Local version of trigram probability calculation for tree statistics."""
    tokens = corpus_text.lower().split()
    
    if len(tokens) < 3:
        return {}
    
    trigrams = [f"{tokens[i]},{tokens[i+1]},{tokens[i+2]}" 
               for i in range(len(tokens)-2)]
    bigrams = [f"{tokens[i]},{tokens[i+1]}" 
              for i in range(len(tokens)-1)]
    
    trigram_freq_dist = Counter(trigrams)
    bigram_freq_dist = Counter(bigrams)
    
    trigram_probabilities = {}
    for trigram, count in trigram_freq_dist.items():
        # Extract first two words from trigram key
        parts = trigram.split(',')
        bigram_key = f"{parts[0]},{parts[1]}"
        bigram_count = bigram_freq_dist.get(bigram_key, 1)
        trigram_probabilities[trigram] = count / bigram_count
    
    return trigram_probabilities


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
        return wrapper
    return decorator


class ReasoningStep(BaseModel):
    explanation: str

class AspectTerm(BaseModel):
    aspectTerm: Union[str, List[str]]
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
        use_enum_values = True

    def dict(self, *args, **kwargs):
        result = super().dict(*args, **kwargs)
        if not result.get('reasoning_steps'):
            result['reasoning_steps'] = ''
        return result

class ImplicitnessPolarityResponse(BaseModel):
    implicitness: List[Implicitness]
    polarity: List[Polarity]


class OptimizedGenDataset:
    """Optimized version using pandas UDFs and native Spark operations instead of broadcast variables + UDFs."""
    
    def __init__(self, args, pre_nlp):
        self.pre_nlp = pre_nlp
        config = Dict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))

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
        self.model = self.config['chat_gpt_model_path']
        self.local_tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GPTQ")

        self.spark_session = (SparkSession.builder
                              .master("local[*]")
                              .appName("OptimizedTiktokComments")
                              .config("spark.sql.adaptive.enabled", "true")
                              .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                              .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
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

        self.pre_nlp_schema = StructType([
            StructField('comments', StringType(), nullable=False),
            StructField('LDA_aspect_prob', ArrayType(StringType()), True),
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
            StructField("LDA_aspect_prob", StringType(), True),
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
        self.base_df, self.raw_input_array = self.initialize_df(self.raw_text_col, self.out_text_col)

    # =========================================================================
    # PHASE 1: OPTIMIZED PANDAS UDFS (3-100x FASTER)
    # https://levelup.gitconnected.com/stop-using-palin-pyspark-udfs-no-one-likes-slow-cars-b5b33ccc80f4
    # =========================================================================
    
    @staticmethod
    @pandas_udf(DoubleType())
    def calc_shannon_entropy_vectorized(text_series: pd.Series) -> pd.Series:
        """Vectorized Shannon entropy calculation using pandas UDF."""
        def compute_entropy(text):
            if pd.isna(text) or not text:
                return 0.0
            tokens = text.split()
            if not tokens:
                return 0.0
            freq_dist = Counter(tokens)
            total_tokens = len(tokens)
            prob_dist = np.array(list(freq_dist.values())) / total_tokens
            # Use numpy for vectorized operations
            entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))  # Add small epsilon to avoid log(0)
            return float(entropy)
        
        return text_series.apply(compute_entropy)

    @staticmethod
    @pandas_udf(DoubleType())
    def calc_mutual_information_vectorized(text_series: pd.Series, prob_dist_col: pd.Series, joint_prob_col: pd.Series) -> pd.Series:
        """Vectorized mutual information calculation using pandas UDF."""
        def compute_mi(text, prob_dist_str, joint_prob_str):
            if pd.isna(text) or not text:
                return 0.0
            
            try:
                # Parse the serialized probability distributions
                prob_dist = json.loads(prob_dist_str) if isinstance(prob_dist_str, str) else prob_dist_str
                joint_prob_dist = json.loads(joint_prob_str) if isinstance(joint_prob_str, str) else joint_prob_str
                
                tokens = text.lower().split()
                mutual_information_score = 0.0
                
                for i in range(len(tokens) - 1):
                    x, y = tokens[i], tokens[i + 1]
                    joint_prob = joint_prob_dist.get(f"{x},{y}", 1e-10)  # Use string key for bigrams
                    marginal_prob_x = prob_dist.get(x, 1e-10)
                    marginal_prob_y = prob_dist.get(y, 1e-10)
                    mutual_information_score += joint_prob * math.log2(joint_prob / (marginal_prob_x * marginal_prob_y + 1e-10))
                
                return float(mutual_information_score)
            except:
                return 0.0
        
        return pd.Series([compute_mi(text, prob, joint) for text, prob, joint in zip(text_series, prob_dist_col, joint_prob_col)])

    @staticmethod
    @pandas_udf(DoubleType())
    def calc_surprisal_vectorized(text_series: pd.Series, trigram_prob_col: pd.Series) -> pd.Series:
        """Vectorized surprisal calculation using pandas UDF."""
        def compute_surprisal(text, trigram_prob_str):
            if pd.isna(text) or not text:
                return 0.0
            
            try:
                trigram_probabilities = json.loads(trigram_prob_str) if isinstance(trigram_prob_str, str) else trigram_prob_str
                words = text.lower().split()
                surprisals = []

                for i in range(2, len(words)):
                    w1, w2, w3 = words[i - 2], words[i - 1], words[i]
                    trigram_key = f"{w1},{w2},{w3}"
                    trigram_prob = trigram_probabilities.get(trigram_key, 1e-10)
                    surprisal = -math.log2(trigram_prob + 1e-10)
                    surprisals.append(surprisal)

                return float(sum(surprisals) / len(surprisals)) if surprisals else 0.0
            except:
                return 0.0
        
        return pd.Series([compute_surprisal(text, prob) for text, prob in zip(text_series, trigram_prob_col)])

    @staticmethod
    @pandas_udf(DoubleType())
    def calc_perplexity_vectorized(text_series: pd.Series, trigram_prob_col: pd.Series) -> pd.Series:
        """Vectorized perplexity calculation using pandas UDF."""
        def compute_perplexity(text, trigram_prob_str):
            if pd.isna(text) or not text:
                return 0.0
            
            try:
                trigram_probabilities = json.loads(trigram_prob_str) if isinstance(trigram_prob_str, str) else trigram_prob_str
                words = text.lower().split()
                N = len(words)
                log_prob_sum = 0.0

                for i in range(2, len(words)):
                    w1, w2, w3 = words[i - 2], words[i - 1], words[i]
                    trigram_key = f"{w1},{w2},{w3}"
                    trigram_prob = trigram_probabilities.get(trigram_key, 1e-10)
                    log_prob_sum += math.log2(trigram_prob + 1e-10)

                avg_log_prob = log_prob_sum / (N - 2) if N > 2 else 0
                perplexity = 2 ** (-avg_log_prob)
                return float(perplexity)
            except:
                return 0.0
        
        return pd.Series([compute_perplexity(text, prob) for text, prob in zip(text_series, trigram_prob_col)])

    # =========================================================================
    # PHASE 2: NATIVE SPARK OPERATIONS (NO BROADCAST + UDF ANTI-PATTERN)
    # =========================================================================
    
    def calc_joint_prob_dist_native(self, corpus_text):
        """Calculate probability distributions using native Spark operations."""
        # Split corpus into tokens and create DataFrame
        tokens_df = self.spark_session.createDataFrame(
            [(token,) for token in corpus_text.lower().split()], 
            ["token"]
        )
        
        # Calculate token frequencies and probabilities
        total_tokens = tokens_df.count()
        prob_dist_df = (tokens_df
                        .groupBy("token")
                        .count()
                        .withColumn("probability", col("count") / total_tokens)
                        .select("token", "probability"))
        
        # Create bigrams for joint probability distribution
        tokens_list = corpus_text.lower().split()
        bigrams = [(tokens_list[i], tokens_list[i+1]) for i in range(len(tokens_list)-1)]
        
        bigrams_df = self.spark_session.createDataFrame(
            bigrams, ["token1", "token2"]
        )
        
        total_bigrams = bigrams_df.count()
        joint_prob_df = (bigrams_df
                         .groupBy("token1", "token2")
                         .count()
                         .withColumn("joint_probability", col("count") / total_bigrams)
                         .withColumn("bigram_key", expr("concat(token1, ',', token2)"))
                         .select("bigram_key", "joint_probability"))
        
        # Convert to dictionaries for serialization
        prob_dist = {row['token']: row['probability'] for row in prob_dist_df.collect()}
        joint_prob_dist = {row['bigram_key']: row['joint_probability'] for row in joint_prob_df.collect()}
        
        return prob_dist, joint_prob_dist

    def calc_trigram_probabilities_native(self, corpus_text):
        """Calculate trigram probabilities using native Spark operations."""
        tokens_list = corpus_text.lower().split()
        
        # Create trigrams
        trigrams = [(tokens_list[i], tokens_list[i+1], tokens_list[i+2]) 
                   for i in range(len(tokens_list)-2)]
        bigrams = [(tokens_list[i], tokens_list[i+1]) 
                  for i in range(len(tokens_list)-1)]
        
        trigrams_df = self.spark_session.createDataFrame(
            trigrams, ["w1", "w2", "w3"]
        )
        
        bigrams_df = self.spark_session.createDataFrame(
            bigrams, ["w1", "w2"]
        )
        
        # Count trigrams and bigrams
        trigram_counts = (trigrams_df
                         .groupBy("w1", "w2", "w3")
                         .count()
                         .withColumnRenamed("count", "trigram_count"))
        
        bigram_counts = (bigrams_df
                        .groupBy("w1", "w2")
                        .count()
                        .withColumnRenamed("count", "bigram_count"))
        
        # Join and calculate conditional probabilities
        trigram_probs = (trigram_counts
                        .join(bigram_counts, ["w1", "w2"])
                        .withColumn("probability", col("trigram_count") / col("bigram_count"))
                        .withColumn("trigram_key", expr("concat(w1, ',', w2, ',', w3)"))
                        .select("trigram_key", "probability"))
        
        # Convert to dictionary for serialization
        trigram_probabilities = {row['trigram_key']: row['probability'] 
                               for row in trigram_probs.collect()}
        
        return trigram_probabilities

    # =========================================================================
    # PHASE 3: WINDOW FUNCTIONS FOR CONTEXTUAL CALCULATIONS
    # =========================================================================
    
    def construct_contextual_scores_optimized(self, df):
        """Use window functions instead of complex UDF with broadcast variables."""
        # Create window partitioned by reply trees
        window_by_tree = Window.partitionBy("Reply to Which Comment").orderBy("Comment Time")
        window_full_tree = Window.partitionBy("Reply to Which Comment")
        
        # Collect comments within each tree using window functions
        df_with_context = (df
                          .withColumn("tree_comments", 
                                    collect_list("Comment").over(window_full_tree))
                          .withColumn("tree_comment_text", 
                                    array_join("tree_comments", " ")))
        
        # Calculate contextual probability distributions for each tree
        # We'll use a more efficient approach by pre-calculating tree-level statistics
        tree_stats = (df_with_context
                     .select("Reply to Which Comment", "tree_comment_text")
                     .distinct()
                     .rdd
                     .map(lambda row: _calculate_tree_stats_standalone(row))
                     .collect())
        
        # Convert tree stats to DataFrame for joining
        tree_stats_df = self.spark_session.createDataFrame(
            tree_stats,
            ["Reply to Which Comment", "tree_prob_dist", "tree_joint_prob", "tree_trigram_prob"]
        )
        
        # Join tree statistics back to main DataFrame
        df_with_stats = df.join(tree_stats_df, "Reply to Which Comment", "left")
        
        # Apply vectorized contextual calculations
        result_df = (df_with_stats
                    .withColumn("contextual_mutual_information_score",
                              OptimizedGenDataset.calc_mutual_information_vectorized(
                                  col("Comment"), 
                                  col("tree_prob_dist"), 
                                  col("tree_joint_prob")))
                    .withColumn("contextual_surprisal",
                              OptimizedGenDataset.calc_surprisal_vectorized(
                                  col("Comment"), 
                                  col("tree_trigram_prob")))
                    .withColumn("contextual_perplexity",
                              OptimizedGenDataset.calc_perplexity_vectorized(
                                  col("Comment"), 
                                  col("tree_trigram_prob")))
                    .drop("tree_comments", "tree_comment_text", "tree_prob_dist", 
                          "tree_joint_prob", "tree_trigram_prob"))
        
        return result_df

    def _calculate_tree_stats(self, row):
        """Helper function to calculate statistics for each comment tree."""
        reply_to_id = row["Reply to Which Comment"]
        tree_text = row["tree_comment_text"]
        
        if not tree_text:
            # Return empty stats for null trees
            return (reply_to_id, "{}", "{}", "{}")
        
        # Calculate distributions for this tree
        prob_dist, joint_prob_dist = self.calc_joint_prob_dist_native_local(tree_text)
        trigram_probabilities = self.calc_trigram_probabilities_native_local(tree_text)
        
        # Serialize to JSON strings for DataFrame storage
        return (
            reply_to_id,
            json.dumps(prob_dist),
            json.dumps(joint_prob_dist),
            json.dumps(trigram_probabilities)
        )

    def calc_joint_prob_dist_native_local(self, corpus_text):
        """Local version of joint probability calculation for tree statistics."""
        tokens = corpus_text.lower().split()
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return {}, {}
        
        # Token probabilities
        freq_dist = Counter(tokens)
        prob_dist = {token: count / total_tokens for token, count in freq_dist.items()}
        
        # Bigram joint probabilities
        if total_tokens < 2:
            return prob_dist, {}
        
        bigrams = [f"{tokens[i]},{tokens[i+1]}" for i in range(total_tokens-1)]
        bigram_freq_dist = Counter(bigrams)
        joint_prob_dist = {bigram: count / (total_tokens - 1) 
                          for bigram, count in bigram_freq_dist.items()}
        
        return prob_dist, joint_prob_dist

    def calc_trigram_probabilities_native_local(self, corpus_text):
        """Local version of trigram probability calculation for tree statistics."""
        tokens = corpus_text.lower().split()
        
        if len(tokens) < 3:
            return {}
        
        trigrams = [f"{tokens[i]},{tokens[i+1]},{tokens[i+2]}" 
                   for i in range(len(tokens)-2)]
        bigrams = [f"{tokens[i]},{tokens[i+1]}" 
                  for i in range(len(tokens)-1)]
        
        trigram_freq_dist = Counter(trigrams)
        bigram_freq_dist = Counter(bigrams)
        
        trigram_probabilities = {}
        for trigram, count in trigram_freq_dist.items():
            # Extract first two words from trigram key
            parts = trigram.split(',')
            bigram_key = f"{parts[0]},{parts[1]}"
            bigram_count = bigram_freq_dist.get(bigram_key, 1)
            trigram_probabilities[trigram] = count / bigram_count
        
        return trigram_probabilities

    # =========================================================================
    # OPTIMIZED INITIALIZATION
    # =========================================================================
    
    def initialize_df(self, raw_text_column, out_text_col):
        """Optimized DataFrame initialization with native Spark operations."""
        # Load base DataFrame
        base_df = (
            self.spark_session.read
            .schema(self.csv_schema)
            .csv(f"{self.input_file_path}", header=True, inferSchema=True)
            .withColumn("Comment Time", from_unixtime(unix_timestamp(col("Comment Time"), "dd/MM/yyyy, HH:mm:ss")))
            .orderBy([asc('Comment'), desc(length(col('Comment')))])
            .withColumn("index", monotonically_increasing_id())
        )
        
        print('Initialize DF:')
        base_df.show(self.batch_size)

        # Set up pre-NLP DataFrame
        self.pre_nlp_df = (
            self.spark_session.createDataFrame(self.pre_nlp, self.pre_nlp_schema)
            .orderBy([asc('comments'), desc(length(col('comments')))])
            .withColumn("index", monotonically_increasing_id())
        ).orderBy(desc(col("index")))

        print('Calculating Scores with Optimized Methods...')

        # Get corpus text for global calculations
        corpus = base_df.selectExpr("collect_list(Comment) as Comment").collect()[0]["Comment"]
        comment_corpus = " ".join(corpus)
        
        # Calculate global probability distributions using native Spark
        prob_dist, joint_prob_dist = self.calc_joint_prob_dist_native(comment_corpus)
        trigram_probabilities = self.calc_trigram_probabilities_native(comment_corpus)
        
        # Serialize distributions as string columns (more efficient than broadcast)
        prob_dist_str = json.dumps(prob_dist)
        joint_prob_str = json.dumps(joint_prob_dist)
        trigram_prob_str = json.dumps(trigram_probabilities)
        
        # Add distribution columns to DataFrame (broadcast alternative)
        base_df = (base_df
                  .withColumn("prob_dist", lit(prob_dist_str))
                  .withColumn("joint_prob_dist", lit(joint_prob_str))
                  .withColumn("trigram_prob_dist", lit(trigram_prob_str)))
        
        # Apply optimized vectorized calculations
        base_df = (base_df
                  .withColumn("shannon_entropy", 
                            OptimizedGenDataset.calc_shannon_entropy_vectorized(col("Comment")))
                  .withColumn("mutual_information_score",
                            OptimizedGenDataset.calc_mutual_information_vectorized(
                                col("Comment"), col("prob_dist"), col("joint_prob_dist")))
                  .withColumn("surprisal",
                            OptimizedGenDataset.calc_surprisal_vectorized(
                                col("Comment"), col("trigram_prob_dist")))
                  .withColumn("perplexity",
                            OptimizedGenDataset.calc_perplexity_vectorized(
                                col("Comment"), col("trigram_prob_dist"))))
        
        # Apply optimized contextual score calculation
        base_df = self.construct_contextual_scores_optimized(base_df)
        
        # Clean up temporary columns
        base_df = base_df.drop("prob_dist", "joint_prob_dist", "trigram_prob_dist")
        
        base_df = base_df.orderBy(desc(col("index")))
        
        print('Optimized Information Scores:')
        base_df.show(self.batch_size)

        # Handle existing processed data
        raw_input_array = base_df.select(raw_text_column).rdd.flatMap(lambda x: x).collect()
        
        if os.path.exists(self.output_file_path):
            self.processed_df = self.spark_session.read.schema(self.final_schema).parquet(f"{self.output_file_path}")
            print('The current parquet df length is: ', self.processed_df.count())
            self.processed_df = self.processed_df.orderBy(desc(col("index")))
            self.processed_df.show(self.batch_size)
            self.processed_ids = self.processed_df.select(out_text_col).distinct().rdd.flatMap(lambda x: x).collect()
            
            if base_df.count() >= self.processed_df.count():
                self.remaining_df = base_df.filter(~base_df[raw_text_column].isin(self.processed_ids))
                self.pre_nlp_df = self.pre_nlp_df.filter(~col('comments').isin(self.processed_ids))
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

    # =========================================================================
    # KEEP EXISTING METHODS THAT DON'T NEED OPTIMIZATION
    # =========================================================================
    
    def extract_text_tokens(self, input_array):
        batch_encoded = self.tokenizer.batch_encode_plus(input_array,
                                                         padding=True,
                                                         max_length=self.config.max_length,
                                                         return_tensors=None)
        print(batch_encoded)
        self.bert_tokens = batch_encoded
        return self.bert_tokens

    def convert_lda_aspects(self, lda_aspects):
        return [json.dumps(aspect) if isinstance(aspect, list) else str(aspect) for aspect in lda_aspects]

    def prep_token_explode(self, batch_df, raw_batch_array):
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

    # Keep all the existing GPT and aspect extraction methods unchanged
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

    @json_error_handler(max_retries=5, delay_seconds=2, spec='Base GPT Prompt')
    @rest_after_run(sleep_seconds=2)
    def prompt_gpt_v2(self, role, prompt, response_format):
        """GPT API call - unchanged"""
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

    def batch_generate_aspect_masks(self, index):
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

    def nlp_batch_for_aspects(self, batch_input, feature_set):
        matching_features = self.pre_nlp[self.pre_nlp['comments'].isin(batch_input)]
        matching_features = matching_features.set_index('comments').reindex(batch_input).reset_index()
        pre_nlp_features = matching_features[feature_set]
        pre_nlp_features_list = pre_nlp_features.to_dict(orient='records')
        assert len(batch_input) == len(pre_nlp_features_list), "The input arrays must have the same length."
        formatted_prompt = ""
        for index, (sentence, features) in enumerate(zip(batch_input, pre_nlp_features_list)):
            formatted_prompt += f"Input sentence {index}: {sentence}\n"
            formatted_prompt += f"Corresponding NLP features {index}: {features}\n\n"
        return formatted_prompt

    def assert_order_v2(self, aspect_terms, batch_input):
        for aspects, input in zip(aspect_terms, batch_input):
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
        polarity = [i.polarity for i in polarity_batch]
        implicitness = [i.implicitness for i in implicitness_batch]

        from pyspark.sql import Row
        
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
        
        isa_schema = StructType([
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
        
        final_train_df = self.spark_session.createDataFrame(rows, isa_schema)

        print('batch_df')
        self.batch_df.show()
        self.batch_df.cache()
        print('final_train_df')
        final_train_df.show()
        final_train_df.cache()

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
                                        col('polarity').alias('labels'))

            train_df.show()

            data_rows = train_df.collect()
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
        while self.remaining_df.count() > 0:
            self.batch_df = self.remaining_df.limit(self.batch_size)
            print('Batch DF')
            self.batch_df.show(self.batch_size)
            raw_batch_array = self.batch_df.orderBy(col("index").desc()).select(self.raw_text_col).rdd.flatMap(lambda x: x).collect()
            batch_index = self.batch_df.orderBy(col("index").desc()).select("index").rdd.flatMap(lambda x: x).collect()

            self.extract_text_tokens(raw_batch_array)
            nlp_feature_set = ['spaCy_tokens', 'POS', 'entities', 'labels', 'negations', 'LDA_aspect_prob']
            batch_nlp = self.nlp_batch_for_aspects(raw_batch_array, nlp_feature_set)
            self.batch_extract_aspects_v2(batch_nlp, nlp_feature_set, 2, raw_batch_array)

            self.batch_df = self.prep_token_explode(self.batch_df, raw_batch_array)

            self.aspects, self.batch_df = self.explode_df_v2(self.batch_df, raw_batch_array, self.raw_text_col,
                                                          self.aspects, 'aspectTerm', dict)

            self.batch_df.cache()
            print('The exploded batch df is now of size:', self.batch_df.count())

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
            self.implicitness, self.polarity = self.batch_extract_polarity_implicitness_v2(batch_nlp, batch_features_2)

            self.processed_batch_df = self.transform_df_v2(raw_text, input_ids, token_type_ids, attention_mask,
                                                        self.aspects, self.aspect_masks, self.polarity, self.implicitness)

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
    debug_out_parquet_path = "./data/gen/debug_train_dataframe.parquet"
    out_pkl_path = './data/gen/Tiktok_Train_Implicit_Labeled_preprocess_finetune.pkl'

    pre_args = parse_arguments(stanza=False, nltk=True, spacy=True)
    preprocessor = NLPTextAnalyzer(args=pre_args)
    comments = preprocessor.read_CSV(raw_file_path)
    nlp_feature_df = preprocessor.construct_nlp_feature_df(comments, 'comments')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='./config/genconfig.yaml', help='config file')
    parser.add_argument('-r', '--raw_file_path', default=raw_file_path)
    parser.add_argument('-s', '--stanza_file_path', default='')
    parser.add_argument('-r_col', '--raw_text_col', default='Comment')
    parser.add_argument('-o', '--out_file_path', default=debug_out_parquet_path)
    parser.add_argument('-o_col', '--out_text_col', default='raw_text')
    parser.add_argument('-of', '--output_format', default='pkl', choices=['xml', 'json', 'pkl'])
    parser.add_argument('-pkl', '--output_pkl_path', default=out_pkl_path)

    args = parser.parse_args()
    
    # Use the optimized version
    gen = OptimizedGenDataset(args=args, pre_nlp=nlp_feature_df)
    gen.run()