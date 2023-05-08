import re
import sys, getopt
import pickle
import pandas as pd
from ordered_set import OrderedSet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


stopWords = ["","a","about","above","after","again","against","all","although","am","an","and","any","app","api","are","aren't","as","at","be","because","been","before","began","being","below","between","both","but","by","came","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","each","even","for","from","gave","given","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","high","him","himself","his","how","however","how's","i","i'd","i'll","i'm","im","i've","if","in","into","is","isn't","it","it's","its","itself","left","let's","kind","made","make","me","more","most","much","mustn't","my","myself","no","nor","of","off","on","one","once","only","or","other","ought","our","ours","ourselves","out","over","seemed","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","time","times","to","too","under","until","up","us","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","will","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]
nonPredWords = ["academic","aidriven","analysis","analytics","attempt","automation","background","basic","asked","boasting","buffer","build","building","built","challenge","choose","code","company","companys","complex","communication","content","consistently","customer","create","creating","dashboard","data","decisions","deliver","delivered","delivering","design","despite","detail","develop","developer","during","ecommerce","educational","engine","end","email","entire","especially","excellent","exceptional","expected","expectations","experience","expertise","fees","few","file","files","final","further","functional","grades","impact","inventory","issues","led","life","login","look","marketing","manage","managed","management","members","mobile","note","occasional","online","open","opportunity","outstanding","own","payment","performance","platform","process","processing","product","programmer","programmers","programming","progress","project","projects","provide","provided","quality","realworld","record","requirements","response","results","resulting","saas","sales","same","satisfaction","scalable","search","security","service","services","skills","solution","stakeholders","streaming","storage","subpar","suggested","support","system","task","tasked","team","teams","tool","truly","turned","ultimately","unfortunately","untimely","update","uploads","user","users","video","warehouse","work","working"]

def fit(infile):
  reviews = []
  allWords = OrderedSet()

  with open(infile) as inf:
    for count, line in enumerate(inf):
      if count == 0: continue # skip header

      review = dict(words=[], stars=0, predStars=0, probStars=(0,0,0))

      # split the line between text and stars
      div = len(line) - 3
      line = line[:div] + '|' + line[div+1:]
      split = line.split('|')

      text = split[0]
      stars = int(split[1])

      review['text'] = text
      review['stars'] = stars

      allWords.add('stars')

      for word in text.split(' '):
      
        # clean the word
        word = word.strip()
        word = word.lower()
        word = re.sub(r'[^\w\s]', '', word)

        if word in stopWords: continue    # don't include stopwords
        if word in nonPredWords: continue # don't include non-predictive words

        allWords.add(word)

        review['words'].append(word)
      reviews.append(review)

  df1star = pd.DataFrame(columns=allWords)
  df2star = pd.DataFrame(columns=allWords)
  df3star = pd.DataFrame(columns=allWords)

  for review in reviews:
    if review['stars'] == 1:
      data1 = [1]
      data2 = [0]
      data3 = [0]
    if review['stars'] == 2:
      data1 = [0]
      data2 = [1]
      data3 = [0]
    if review['stars'] == 3:
      data1 = [0]
      data2 = [0]
      data3 = [1]

    for word in allWords[1:]:
      if word in review['words']: 
        data1.append(1)
        data2.append(1)
        data3.append(1)
      else:
        data1.append(0)
        data2.append(0)
        data3.append(0)

    df1star.loc[len(df1star.index)] = data1
    df2star.loc[len(df2star.index)] = data2
    df3star.loc[len(df3star.index)] = data3

  x_train, y_train = df1star.drop(columns='stars'), df1star['stars']
  logit_1star = LogisticRegression(solver = 'newton-cg', max_iter= 150).fit(x_train,y_train)
  pred_1star = logit_1star.predict_proba(x_train)

  x_train, y_train = df2star.drop(columns='stars'), df2star['stars']
  logit_2star = LogisticRegression(solver = 'newton-cg', max_iter= 150).fit(x_train,y_train)
  pred_2star = logit_2star.predict_proba(x_train)

  x_train, y_train = df3star.drop(columns='stars'), df3star['stars']
  logit_3star = LogisticRegression(solver = 'newton-cg', max_iter= 150).fit(x_train,y_train)
  pred_3star = logit_3star.predict_proba(x_train)


  print()
  print('stars pred   prob *        prob **       prob ***       words')
  for count, review in enumerate(reviews):
    review['probStars'] = ("{:.8f}".format(pred_1star[count][1]),"{:.8f}".format(pred_2star[count][1]),"{:.8f}".format(pred_3star[count][1]))

    # determine the logit model with the highest probability
    pred = max(review['probStars'])
    review['predStars'] = review['probStars'].index(pred) + 1

    print('  ' + str(review['stars']), '    ' + str(review['predStars']), '  ' + str(review['probStars']), review['words'])

  # print results
  correct = 0
  wrong = 0
  for review in reviews:
    if review['stars'] == review['predStars']:
      correct += 1
    else:
      wrong += 1

  accuracy = correct/(correct+wrong)*100

  print()
  print('correct:',correct, '\nwrong:',wrong)
  print('accuracy: ' + "{:.2f}".format(accuracy) + '%')
  print()

  # save the fit models
  pickle.dump(allWords,    open("allWords.pkl","wb"))
  pickle.dump(logit_1star, open("logitModel1.pkl","wb"))
  pickle.dump(logit_2star, open("logitModel2.pkl","wb"))
  pickle.dump(logit_3star, open("logitModel3.pkl","wb"))

def pred(infile):
  reviews = []

  allWords = pickle.load(open( "allWords.pkl","rb"))
  logit_1star = pickle.load(open( "LogitModel1.pkl","rb"))
  logit_2star = pickle.load(open( "LogitModel2.pkl","rb"))
  logit_3star = pickle.load(open( "LogitModel3.pkl","rb"))

  with open(infile) as inf:
    for count, line in enumerate(inf):
      line = line.strip()
      if not line: continue   # skip blank lines
      if count == 0: continue # skip the header

      review = dict(text=line, words=[], predStars=0, probStars=(0,0,0))

      for word in line.split(' '):
      
        # clean the word
        word = word.strip()
        word = word.lower()
        word = re.sub(r'[^\w\s]', '', word)

        if word in stopWords: continue    # don't include stopwords
        if word in nonPredWords: continue # don't include non-predictive words

        review['words'].append(word)
      reviews.append(review)

  allWords.remove('stars')
  df = pd.DataFrame(columns=allWords)

  for review in reviews:
    data=[]

    for word in allWords:
      if word in review['words']: 
        data.append(1)
      else:
        data.append(0)

    df.loc[len(df.index)] = data

  pred_1star = logit_1star.predict_proba(df)
  pred_2star = logit_2star.predict_proba(df)
  pred_3star = logit_3star.predict_proba(df)

  print()
  print('stars  prob 1 star   prob 2 stars  prob 3 stars   words')
  for count, review in enumerate(reviews):
    review['probStars'] = ("{:.8f}".format(pred_1star[count][1]),"{:.8f}".format(pred_2star[count][1]),"{:.8f}".format(pred_3star[count][1]))

    # determine the logit model with the highest probability
    pred = max(review['probStars'])
    review['predStars'] = review['probStars'].index(pred) + 1

    print('  ' + str(review['predStars']), '  ' + str(review['probStars']), review['words'])

  # print the results
  print()
  print('stars  review')
  for review in reviews:
    print('  ' + str(review['predStars']) + '    ' + review['text'])
  print()


def main(argv):
  infile = ''
  mode = ''

  try:
    opts, args = getopt.getopt(argv,"hi:m:",["inf=","mode="])
  except getopt.GetoptError:
    print('unrecognized parameter')
    print ('useage: bayes.py -i <inputfile> -m <mode: fit or pred>')
    print()
    sys.exit()

  for opt, arg in opts:
    if opt == '-h':
      print ('bayes.py -i <inputfile> -m <mode: fit or pred>')
      sys.exit()
    elif opt in ("-i", "--inf"):
      infile = arg
    elif opt in ("-m", "--mode"):
      if arg not in ['fit', 'pred']:
        print('unknown mode')
        print ('useage: bayes.py -i <inputfile> -m <mode: fit or pred>')
        print()
        sys.exit()
      else:  
        mode = arg

  if not infile or not mode: 
    print('missing parameter')
    print ('useage: bayes.py -i <inputfile> -m <mode: fit or pred>')
    print()
    sys.exit()

  if mode == 'fit': fit(infile)
  elif mode == 'pred': pred(infile)


if __name__ == "__main__":
  main(sys.argv[1:])


