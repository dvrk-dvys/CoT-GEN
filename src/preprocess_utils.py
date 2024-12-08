import argparse
import stanza
import pandas as pd
from typing import List
from stanza.models.common.doc import Document
from stanza_batch import batch
from src.utils import runtime
#import toma
import nltk
from nltk.corpus import stopwords
import spacy

from gensim import corpora
from gensim.models import Word2Vec, LdaModel


class NLPTextAnalyzer:
    def __init__(self, args):
        if args.stanza:
            stanza.download('en')
            self.stanza_pipe = stanza.Pipeline('en', use_gpu=True)
        if args.spacy:
           self.nlp = spacy.load("en_core_web_lg")
        if args.nltk:
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))

    def gen_stanza_analysis(self, text):
        doc = self.stanza_pipe(text)
        return doc

    def read_CSV(self, path):
        df = pd.read_csv(path)
        column = df['Comment']
        return column

    def read_Parquet(self, path):
        df = pd.read_parquet(path)
        column = df['Comment']
        return column

    def batch_process_fixed(self, comments: List[str], batch_size: int = 5) -> List[stanza.Document]:
        all_docs = []
        for i in range(0, len(comments), batch_size):
            batch_comments = comments[i:i+batch_size]
            in_docs = [stanza.Document([], text=d) for d in batch_comments]
            out_docs = self.stanza_pipe(in_docs)
            all_docs.extend(out_docs)
        return all_docs
    #def run_batch(self, batch_size: int, nlp, texts: List[str]) -> List[Document]:
    #    return self.gen_analysis(texts, batch_size)

    #def batch_process_toma(self, comments: List[str]) -> List[Document]:
    #    return toma.batch(self.run_batch, 500, self.nlp, comments)

    @runtime
    def stanza_processor(self, comments):
        data = []
        processed_docs = self.batch_process_fixed(comments, batch_size=10)
        #processed_docs = self.batch_process_toma(comments)

        for doc, comment in zip(processed_docs, comments):
            text_list, lemma_list, upos_list, head_list, deprel_list, ner_list = [], [], [], [], [], []
            for sentence in doc.sentences:
                text_list.extend([word.text for word in sentence.words])
                lemma_list.extend([word.lemma for word in sentence.words])
                upos_list.extend([word.upos for word in sentence.words])
                head_list.extend([word.head for word in sentence.words])
                deprel_list.extend([word.deprel for word in sentence.words])
                ner_list.extend([getattr(word, 'ner', None) for word in sentence.words])
            data.append({
                'sentence': comment,
                'text_list': text_list,
                'lemma_list': lemma_list,
                'upos_list': upos_list,
                'head_list': head_list,
                'deprel_list': deprel_list,
                'ner_list': ner_list
            })
        return data

    def old_nlp_processor(self, comments):
        data = []
        for comment in comments:
            doc = self.gen_analysis(comment)
            text_list, lemma_list, upos_list, head_list, deprel_list, ner_list = [], [], [], [], [], []

            for sentence in doc.sentences:
                text_list.extend([word.text for word in sentence.words])
                lemma_list.extend([word.lemma for word in sentence.words])
                upos_list.extend([word.upos for word in sentence.words])
                head_list.extend([word.head for word in sentence.words])
                deprel_list.extend([word.deprel for word in sentence.words])
                ner_list.extend([getattr(word, 'ner', None) for word in sentence.words])

                data.append({
                    'sentence': comment,
                    'text_list': text_list,
                    'lemma_list': lemma_list,
                    'upos_list': upos_list,
                    'head_list': head_list,
                    'deprel_list': deprel_list,
                    'ner_list': ner_list
                })
        return data

    def extract_aspects_lda_sentence(self, comments, num_topics=2, num_words=1):
        aspects = []

        for comment in comments:
            # Tokenize and filter the sentence
            tokens = [word for word in nltk.word_tokenize(comment.lower()) if
                      word.isalnum() and word not in self.stop_words]

            # Handle edge case where there might not be enough tokens to form topics
            if len(tokens) == 0:
                #return []
                aspects.append([])
                continue

            # Create a dictionary and a corpus for LDA
            dictionary = corpora.Dictionary([tokens])
            corpus = [dictionary.doc2bow(tokens)]

            # Train LDA model on the single sentence tokens
            try:
                lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
            except:
                print()
            # Extract topics from the LDA model
            topics = lda_model.print_topics(num_words=num_words)
            aspects.append(topics)
        return aspects

    def extract_aspects_lda_document(self, documents, num_topics=5, num_words=5):
        texts = [[word for word in nltk.word_tokenize(doc.lower()) if word.isalnum() and word not in self.stop_words] for doc
                 in documents]

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

        topics = lda_model.print_topics(num_words=num_words)

        # for topic in topics:
        # print(topic)
        return topics

    def extract_spaCy_features(self, doc):
        artifacts = {
            'spaCy_tokens': [],
            'POS': [],
            'POS_tags': [],
            'dependencies': [],
            'lemmas': [],
            'heads': [],
            'negations': [],
            'entities': [],
            'labels': [],
            'sentences': []
        }

        for token in doc:
            artifacts['spaCy_tokens'].append(token.text)
            artifacts['POS'].append(token.pos_)
            artifacts['POS_tags'].append(token.tag_) #    question_tags = {'WDT', 'WP', 'WP$', 'WRB'}
            artifacts['dependencies'].append(token.dep_)
            artifacts['lemmas'].append(token.lemma_)
            artifacts['heads'].append(token.head.text)
            if token.dep_ == 'neg':
                artifacts['negations'].append(token.head.text)
        for ent in doc.ents:
            artifacts['entities'].append(ent.text)
            artifacts['labels'].append(ent.label_)
        for span in doc.sents:
            artifacts['sentences'].append(span.text)
        return artifacts

    def batch_preprocess_text(self, input_texts):
        self.spaCy_features = {
            'spaCy_tokens': [],
            'POS': [],
            'POS_tags': [],
            'dependencies': [],
            'lemmas': [],
            'heads': [],
            'negations': [],
            'entities': [],
            'labels': [],
            'sentences': []
        }

        for doc in self.nlp.pipe(input_texts):
            features = self.extract_spaCy_features(doc)
            self.spaCy_features['spaCy_tokens'].append(features['spaCy_tokens'])
            self.spaCy_features['POS'].append(features['POS'])
            self.spaCy_features['POS_tags'].append(features['POS_tags'])
            self.spaCy_features['dependencies'].append(features['dependencies'])
            self.spaCy_features['lemmas'].append(features['lemmas'])
            self.spaCy_features['heads'].append(features['heads'])
            self.spaCy_features['negations'].append(features['negations'])
            self.spaCy_features['entities'].append(features['entities'])
            self.spaCy_features['labels'].append(features['labels']) #labels are the label of the entity
            self.spaCy_features['sentences'].append(features['sentences'])
        return self.spaCy_features

    def construct_nlp_feature_df(self, raw_text, col_name):
        aspect_col = self.extract_aspects_lda_sentence(raw_text, num_topics=2, num_words=5)
        spacy_features_col = self.batch_preprocess_text(raw_text)
        raw_text_df = pd.DataFrame({col_name: raw_text})
        aspects_df = pd.DataFrame({'LDA_aspect_prob': aspect_col})
        aspects_df['LDA_aspect_prob'] = aspects_df['LDA_aspect_prob'].apply(lambda x: [str(item) for item in x])
        spacy_features_df = pd.DataFrame(spacy_features_col)
        pre_nlp_df = pd.concat([raw_text_df, aspects_df, spacy_features_df], axis=1)
        #print(pre_nlp_df)
        return pre_nlp_df

    def save(self, data, csv_path, parquet_path):
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)

def parse_arguments(stanza=True, nltk=False, spacy=False):
    parser = argparse.ArgumentParser(description="NLP Text Analyzer Configuration")
    parser.add_argument('--stanza', action='store_true', default=stanza, help='Enable Stanza NLP processing')
    parser.add_argument('--nltk', action='store_true', default=nltk, help='Enable NLTK processing')
    parser.add_argument('--spacy', action='store_true', default=spacy, help='Enable spaCy NLP processing')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments(stanza=False, nltk=True, spacy=True)
    analyzer = NLPTextAnalyzer(args=args)
    raw_tt = '/Users/jordanharris/Code/THOR-GEN/data/raw/TTCommentExporter-7226101187500723498-201-comments.csv'
    comments = analyzer.read_CSV(raw_tt)
    nlp_feature_df = analyzer.construct_nlp_feature_df(comments, 'comments')
    print(nlp_feature_df)


    train_parquet_path = "/Users/jordanharris/Code/THOR-GEN/data/gen/keep_train_dataframe.parquet"
    csv_output_path = "/Users/jordanharris/Code/THOR-GEN/data/gen/stanza-7226101187500723498-201.parquet.csv"
    parquet_output_path = "/Users/jordanharris/Code/THOR-GEN/data/gen/stanza-7226101187500723498-201.parquet"

    # Example usage for a Parquet file
    #comments = analyzer.read_Parquet(train_parquet_path)
    #data = analyzer.stanza_processor(comments)
    #analyzer.save(data, csv_output_path, parquet_output_path)
    #analyzer.process_and_save(comments, csv_output_path, parquet_output_path)
    print()

