import argparse
import pickle
import os

import yaml
from attrdict import AttrDict

import transformers
from transformers import TFRobertaModel
from transformers import AutoTokenizer
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, upper, left, rank
from pyspark.sql.functions import lit, udf, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, BinaryType, BooleanType

from openai import OpenAI
import json

from src.utils import prompt_direct_inferring, prompt_direct_inferring_masked, prompt_for_aspect_inferring


# indexs 0, 1, 6, 8 from
# laptops_test_gold_pkl_file = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/laptops/Laptops_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'
# indexs 0, 2, from
# laptops_train_gold_pkl_file = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/laptops/Laptops_Train_v2_Implicit_Labeled_preprocess_finetune.pkl'
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

class genDataset:
    def __init__(self, args):
        config = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        for k, v in vars(args).items():
            setattr(config, k, v)
        self.config = config
        self.config['openai_token'] = os.getenv("OPENAI_API_KEY")

        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_path)

        # self.df = pd.read_csv(args.raw_file)

        self.spark_session = (SparkSession.builder
                              .master("local[*]")
                              .appName("TiktokComments")
                              .getOrCreate())

        self.input_file = args.raw_file_path
        self.id_text_token_df, self.raw_input_array = self.intialize_df()

    def intialize_df(self):
        schema = StructType([
            StructField("Comment id", StringType(), True),
            StructField("Comment", StringType(), True),
            # StructField("bert_tokens", ArrayType(IntegerType()), True)
        ])

        id_text_token_df = (self.spark_session.read.csv(f"{self.input_file}", header=True, schema=schema)
                            .withColumnRenamed("Comment ID", "tt_comment_id")
                            .withColumnRenamed("Comment", "raw_text")
                            .withColumn("index", monotonically_increasing_id())
                            )

        id_text_token_df.show()  # This will print the first 20 rows in the DataFrame.

        # RDD stands for Resilient Distributed Dataset, which is a fundamental data structure in Apache Spark.It's a fault-tolerant collection of elements that can be operated on in parallel across a cluster of computers.
        raw_input_array = id_text_token_df.select("raw_text").rdd.flatMap(lambda x: x).collect()
        return id_text_token_df, raw_input_array

        # self.raw_input_array = None

    """
    The choice between using BERT or T5 (like flan-t5-base) largely depends on the specific task and the way the model was fine-tuned or trained. Both BERT and T5 are powerful transformer models but are designed with different architectures and objectives:
    1: BERT (Bidirectional Encoder Representations from Transformers) is designed to understand the context of words in a sentence by considering the words that come before and after the target word. It's primarily used for tasks like Named Entity Recognition (NER), sentiment analysis, and question answering.
    2: T5 (Text-to-Text Transfer Transformer) takes a different approach by treating every NLP problem as a text-to-text problem, meaning it converts all NLP tasks into a text-to-text format. This model is versatile and can be used for a variety of tasks, such as translation, summarization, question answering, and more.
    """

    def extract_text_tokens(self, id_text_token_df):
        # # RDD stands for Resilient Distributed Dataset, which is a fundamental data structure in Apache Spark.It's a fault-tolerant collection of elements that can be operated on in parallel across a cluster of computers.
        # self.raw_input_array = id_text_token_df.select("raw_texts").rdd.flatMap(lambda x: x).collect()
        batch_encoded = self.tokenizer.batch_encode_plus(self.raw_input_array, padding=True,
                                                         max_length=self.config.max_length, return_tensors=None)
        print(batch_encoded)
        self.tokens = batch_encoded
        return self.tokens

    def prompt_gpt(self, role, prompt):
        """
        !!!!!!!THIS IS PAID!!!!!!!
        """
        GPTclient = OpenAI()

        completion = GPTclient.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ]
        )
        response = completion.choices[0].message.content

        try:
            response = json.loads(response)
            print(response)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
        return response

    def generate_aspect_mask(self, sentence_tokens, aspect_tokenized):
        mask = [0] * len(sentence_tokens)
        aspect_len = len(aspect_tokenized)
        for i in range(len(sentence_tokens) - aspect_len + 1):
            if sentence_tokens[i:i + aspect_len] == aspect_tokenized:
                for j in range(i, i + aspect_len):
                    mask[j] = 1
        return mask

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

    def batch_extract_aspects(self):
        new_context = f'Given these sentences "{self.raw_input_array}", '
        prompt = new_context + f'which words or phrases are the aspect terms?'
        role = (
            "You are operating as a system that, given a list of sentences, you will identify the core word or phrase in each sentence that are the aspect term or target term of a sentence "
            "that other words in the sentence point to and augment. They do so either implicitly or explicitly. Return the found aspect term(s) as a JSON array, where each entry is an object. "
            "Each object should have a key 'aspectTerm' that either contains the aspect term or is set to 'NONE' if no aspect is found. "
            "Ensure each object in the array corresponds to the index of the sentence provided.")
        self.aspects = self.prompt_gpt(role, prompt)
        self.aspect_masks = []
        for i, a in enumerate(self.aspects):
            # aspect_tokens = self.tokenizer.tokenize(a['aspectTerm'])
            encoded_aspect_tokens = self.tokenizer.encode(a['aspectTerm'], add_special_tokens=False)
            self.aspect_masks.append(self.generate_aspect_mask(self.tokens.data['input_ids'][i], encoded_aspect_tokens))
        return self.aspects, self.aspect_masks

    def extract_polarity_implicitness(self, aspects):
        new_context = f'Given the sentence "{self.raw_input_array}", and this/these aspect term(s),"{aspects}'
        prompt = new_context + f'determine the polairty (positive, negative or neutral) of aspect term and if it is explicitely or implicitely expressed with respect to the whole sentence?'
        role = (
            "You are operating as a system that, given a sentence & aspect term pair, you will identify the sentiment/polarity of the aspect term within the context of the given sentence."
            "Polarity is either positive, negative or neutral {0, 1, 2}. Then determine if the expression is implicit or explicit (True or False)."
            "Return each found polarity (numeric) and implicitness (boolean) as a tuple (numeric, boolean) in a list where each entry corresponds to the input index of the sentence-aspect pair."
            "If the value of the aspect is NONE then place 'NONE' at the index of that input.")
        self.polarity_implicitness = self.prompt_gpt(role, prompt)
        return self.polarity_implicitness

    def batch_extract_polarity_implicitness(self, aspects):
        new_context = f'Given these sentences "{self.raw_input_array}" and aspect term pairs,"{aspects}"'
        prompt = new_context + f'determine the polairty (positive, negative or neutral) of aspect term and if it is explicitely or implicitely expressed with respect to the whole sentence?'
        role = (
            "You are operating as a system that, given a list of sentence & aspect term pairs, you will identify the sentiment/polarity of the aspect term within the context of the given sentence."
            "Polarity is either positive (0), negative (1) or neutral (2). Also, determine if the expression is implicit or explicit (True or False)."
            "Return the results in a JSON format, where each entry is an object with 'polarity' and 'implicitness' keys. Each entry corresponds to the input index of the sentence-aspect pair."
            "For entries with no applicable aspect, return an object with the value {'polarity': 'NONE', 'implicitness': 'NONE'}. "
            "Ensure the output contains only this JSON array and no additional text.")
        self.polarity_implicitness = self.prompt_gpt(role, prompt)
        return self.polarity_implicitness

    def transform_df(self, id_text_token, aspect, aspect_mask, polarity_implicitness):
        aspect_terms = [i['aspectTerm'] for i in aspect]

        implicitness = [i['implicitness'] for i in polarity_implicitness]
        polarity = [i['polarity'] for i in polarity_implicitness]

        token_ids = id_text_token.data['input_ids']
        token_type_ids = id_text_token.data['token_type_ids']
        attention_masks = id_text_token.data['attention_mask']

        rows = [
            Row(
                aspect=aspect_terms[i],
                aspect_mask=aspect_mask[i],
                token_ids=token_ids[i],
                token_type_ids=token_type_ids[i],
                attention_mask=attention_masks[i],
                implicitness=implicitness[i],
                polarity=polarity[i],
                raw_text=self.raw_input_array[i]
            )
            for i in range(len(aspect_terms))
        ]

        df = self.spark_session.createDataFrame(rows)

        df.show()

        final_df = df.alias('a').join(self.id_text_token_df.alias('b'), col('a.raw_text') == col('b.raw_text'), "inner") \
            .select('a.*',
                    'b.tt_comment_id',
                    'b.index')

        return final_df

    def run(self):
        self.extract_text_tokens(self.id_text_token_df)
        self.batch_extract_aspects()
        self.batch_extract_polarity_implicitness(self.aspects)
        return self.transform_df(self.tokens, self.aspects, self.aspect_masks, self.polarity_implicitness)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='./config/genconfig.yaml', help='config file')
    parser.add_argument('-i', '--raw_file_path', default='/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/raw/raw_dev.csv')
    parser.add_argument('-of', '--output_format', default='json', choices=['xml', 'json'])
    args = parser.parse_args()
    # config = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/config/genconfig.yaml'
    # df = pd.read_csv('/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/raw/TTCommentExporter-7352614724489547051-127-comments.csv')
    gen = genDataset(args=args)
    result_df = gen.run()
    result_df.show()


    train_df = result_df.select(col('raw_text').alias('raw_texts'),
                                col('aspect').alias('raw_aspect_terms'),
                                col('token_ids').alias('bert_tokens'),
                                col('aspect_mask').alias('aspect_mask'),
                                col('implicitness').alias('implicits'),
                                col('polarity').alias('labels'))
    train_df.show()

    data = train_df.collect()

    # with open('/data/gen/Tiktok_Train_Implicit_Labeled_preprocess_finetune.pkl', 'w') as file:
    #     pickle.dump(data, file)

    # remove the token_ids
    # result_df.write.csv(path='/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/gen/Tiktok_Train.csv', mode='overwrite', header=True)

