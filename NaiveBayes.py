import pyspark
from pyspark.sql.functions import col, count, when, isnull, udf
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, FloatType, DoubleType, ArrayType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes  # Import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import html
import emo_unicode
from emo_unicode import *
import nltk
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re

lemmatizer = WordNetLemmatizer() # set lemmatizer
stemmer = PorterStemmer() # set stemmer
stopwords_list = stopwords.words('english')
allowedWordTypes = ["J","R","V","N"]

# title abbrevation dict

#source: https://www.kaggle.com/code/nmaguette/up-to-date-list-of-slangs-for-text-preprocessing
abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bah" : "frustration",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "boooo" : "displeasure",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "fml" : "fuck my life",
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "hun" : "honey",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "ott" : "over the top",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pedos" : "pedophile",
    "perma" : "permanent",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pmsl" : "pissing myself laughing",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w00t" : "joy",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

spark = SparkSession.builder.config('spark.driver.memory', '4g').appName('TwitterSentimentAnalysis').getOrCreate()

# data preprocess
metadf = (((((metadf.withColumnRenamed("_c0", "Polarity")
              .withColumnRenamed("_c1", "TweetID"))
             .withColumnRenamed("_c2", "Date"))
            .withColumnRenamed("_c3", "QueryFlag"))
           .withColumnRenamed("_c4", "User"))
          .withColumnRenamed("_c5", "TweetText"))
missing_count = metadf.select([count(when(isnull(c), c)).alias(c) for c in metadf.columns]).collect()
print(missing_count)
duplicates = metadf.groupBy(metadf.columns).count().filter("count > 1")
duplicates.show()

# check schema
metadf.printSchema() 

print(f"There are {metadf.count()} rows and  {len(metadf.columns)} columns in the dataset.")

cols_to_drop= ("TweetID","Date","QueryFlag","User")
metadf = metadf.drop(*cols_to_drop)

metadf = metadf.dropDuplicates()
print(f"Number of rows in the dataframe after dropping the duplicates: {metadf.count()}")
metadf.printSchema()
metadf.show(50, truncate = False)
# check the number of distinct labels and the respective counts 
metadf.groupBy('Polarity').count().orderBy('count').show()
# turn text polarity into double type

def polarity_map(value):
    if value == 4:
        return 1.0  # Positive
#     elif value == 2:
#         return 1.0  # Neutral
    else:
        return 0.0  # Negative

polarity_udf = udf(polarity_map, DoubleType())
metadf = metadf.withColumn("label", polarity_udf(metadf["Polarity"]))

# check the number of distinct labels and the respective counts 
metadf.groupBy('label').count().orderBy('count').show()

# https://github.com/Deffro/text-preprocessing-techniques/blob/master/techniques.py
def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text

def replaceURL(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text
def replaceAtUser(text):
    """ Replaces "@user" with "atUser" """
    text = re.sub('@[^\s]+','atUser',text)
    return text
def removeHashtagInFrontOfWord(text):
    """ Removes hastag in front of a word """
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", ' multiExclamation ', text)
    return text

def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", ' multiQuestion ', text)
    return text

def replaceMultiStopMark(text):
    """ Replaces repetitions of stop marks """
    text = re.sub(r"(\.)\1+", ' multiStop ', text)
    return text

contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
def replaceContraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

# tokenizer = RegexpTokenizer(r'[A-Za-z0-9_@#\']+')
tokenizer = RegexpTokenizer(r'[A-Za-z0-9_\']+')

# tokenizer = RegexpTokenizer(r'@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+')
def tokenize(s):
    full_tokens = []
    final_tokens = []    
    tokens = tokenizer.tokenize(s)
    tokens = [abbreviations.get(word, word) for word in tokens]
    for t in tokens:
        full_tokens=full_tokens+t.split()
    tagged = nltk.pos_tag(full_tokens)
    for w in tagged:
        if (w[1][0] in allowedWordTypes and w[0] not in stopwords_list):
            final_word = w[0]
#             final_word = lemmatizer.lemmatize(final_word)
            final_word = stemmer.stem(final_word)       
            final_tokens.append(final_word)
    final_text = " ".join(final_tokens)
    return final_tokens

def preprocess(text):
    # convert all to lower case
    # replace urls
    # replace special characters
#     text_cleaned = re.sub(replace_url_re, 'URL', str(text).lower()).strip()
    text_cleaned = replaceURL(str(text).lower()).strip()
    text_cleaned = html.unescape(text_cleaned)
    text_cleaned = removeUnicode(text_cleaned)

    for emoji,feeling  in EMOTICONS_EMO.items():
        text_cleaned = text_cleaned.replace(emoji,feeling)
    text_cleaned=" ".join(abbreviations.get(word, word) for word in text_cleaned.split())
    text_cleaned = replaceAtUser(text_cleaned)
    text_cleaned = removeHashtagInFrontOfWord(text_cleaned)
    text_cleaned = replaceMultiExclamationMark(text_cleaned)
    text_cleaned = replaceMultiQuestionMark(text_cleaned)
    text_cleaned = replaceMultiStopMark(text_cleaned)
#     
    text_cleaned = replaceContraction(text_cleaned)
    
    text_tokenized = tokenize(text_cleaned)
#     text_tokenized = [abbreviations.get(word, word) for word in text_tokenized]
    return text_tokenized

metadf.show(10, truncate = False)

metadf.printSchema()

# word2vec process
word2Vec = Word2Vec(vectorSize=100, minCount=5, inputCol="tokens_cleaned", outputCol="features")

# Naive Bayes classifier
nb = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0, modelType="multinomial")

# Create a new pipeline with Naive Bayes
pipeline = Pipeline(stages=[word2Vec, nb])

# Train the Naive Bayes model
model = pipeline.fit(train)

# Make predictions on the test data
predictions = model.transform(test)

# Print the schema and show the predictions
predictions.printSchema()
predictions.show(5, truncate=False)

# Evaluate the Naive Bayes model
predictionAndLabels = predictions.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

evaluator = MulticlassClassificationEvaluator(metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"The f1_score of Naive Bayes model is: {f1_score}")

evaluator = MulticlassClassificationEvaluator(metricName="weightedPrecision")
weightedPrecision = evaluator.evaluate(predictionAndLabels)
print(f"The testing weightedPrecision of Naive Bayes model is: {weightedPrecision}")

evaluator = MulticlassClassificationEvaluator(metricName="weightedRecall")
weightedRecall = evaluator.evaluate(predictions)
print(f"The testing weightedRecall of Naive Bayes model is: {weightedRecall}")
