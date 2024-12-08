from sparknlp.base import *
import pyspark as spark
from sparknlp.annotator import T5Transformer

#https://medium.com/spark-nlp/spark-nlp-101-document-assembler-500018f5f6b5
#https://sparknlp.org/2023/01/31/t5_informal_en.html

documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_informal", "en") \
    .setInputCols("document") \
    .setOutputCol("answers")

pipeline = Pipeline(stages=[documentAssembler, t5])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)