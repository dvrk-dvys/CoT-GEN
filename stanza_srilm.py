import stanza
import pandas as pd


class NLPTextAnalyzer:
    def __init__(self):
        stanza.download('en')
        self.nlp = stanza.Pipeline('en')

    def gen_analysis(self, text):
        doc = self.nlp(text)
        return doc

    def read_CSV(self, path):
        df = pd.read_csv(path)
        column = df['Comment']
        return column

    def read_Parquet(self, path):
        df = pd.read_parquet(path)
        column = df['Comment']
        return column
    def process_and_save(self, comments, csv_path, parquet_path):
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

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)


if __name__ == '__main__':
    analyzer = NLPTextAnalyzer()
    train_parquet_path = "./data/gen/train_dataframe.parquet"
    csv_output_path = "./data/gen/stanza-7226101187500723498-201.parquet.csv"
    parquet_output_path = "./data/gen/stanza-7226101187500723498-201.parquet"


    # Example usage for a Parquet file
    comments = analyzer.read_Parquet(train_parquet_path)
    analyzer.process_and_save(comments, csv_output_path, parquet_output_path)
    print()

