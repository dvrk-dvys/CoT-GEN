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
    def write_doc_to_parquet(self, path):
        None


if __name__ == '__main__':
    analyzer = NLPTextAnalyzer()
    train_parquet_path = "/Users/joergbln/Desktop/JAH/Code/THOR-GEN/data/gen/train_dataframe.parquet"

    # Example usage for a Parquet file
    comments = analyzer.read_Parquet(train_parquet_path)
    for comment in comments:
        doc = analyzer.gen_analysis(comment)
        print(doc)


    # Example usage for a CSV file
    #comments = analyzer.read_CSV('path_to_your_file.csv')
    #for comment in comments:
    #    analyzer.gen_analysis(comment)

