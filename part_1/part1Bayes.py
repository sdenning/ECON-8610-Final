import re
import sys, getopt
import pickle

stopWords = ["","a","about","above","after","again","against","all","although","am","an","and","any","app","api","are","aren't","as","at","be","because","been","before","began","being","below","between","both","but","by","came","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","each","even","for","from","gave","given","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","high","him","himself","his","how","however","how's","i","i'd","i'll","i'm","im","i've","if","in","into","is","isn't","it","it's","its","itself","left","let's","kind","made","make","me","more","most","much","mustn't","my","myself","no","nor","of","off","on","one","once","only","or","other","ought","our","ours","ourselves","out","over","seemed","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","time","times","to","too","under","until","up","us","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","will","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]
nonPredWords = ["ability","academic","aidriven","analysis","analytics","attempt","automation","background","basic","asked","boasting","buffer","build","building","built","challenge","choose","code","company","companys","complex","communication","content","consistently","customer","create","creating","dashboard","data","decisions","deliver","delivered","delivering","design","despite","detail","develop","developer","during","ecommerce","educational","engine","end","email","entire","especially","excellent","exceptional","expected","expectations","experience","expertise","fees","few","file","files","final","further","functional","grades","impact","inventory","issues","lack","led","life","login","look","marketing","manage","managed","management","members","mobile","note","occasional","online","open","opportunity","outstanding","own","payment","performance","platform","process","processing","product","programmer","programmers","programming","progress","project","projects","provide","provided","quality","realworld","record","requirements","response","results","resulting","saas","sales","same","satisfaction","scalable","search","security","service","services","skills","solution","stakeholders","streaming","storage","subpar","suggested","support","system","task","tasked","team","teams","tool","truly","turned","ultimately","unfortunately","untimely","update","uploads","user","users","video","warehouse","work","working"]

def fit(infile):
  reviews = []
  allWords = set()

  BOW = {1:{},2:{},3:{}}

  with open(infile) as inf:
    for count, line in enumerate(inf):
      if count == 0: continue # skip header

      review = dict(words=[], stars=0, predStars=0, bowStars=(0,0,0))

      # split the line between text and stars
      div = len(line) - 3
      line = line[:div] + '|' + line[div+1:]
      split = line.split('|')

      text = split[0]
      stars = int(split[1])

      review['text'] = text
      review['stars'] = stars

      # create a bag of words for each star
      for word in text.split(' '):
      
        # clean the word
        word = word.strip()
        word = word.lower()
        word = re.sub(r'[^\w\s]', '', word)

        if word in stopWords: continue    # don't include stopwords
        if word in nonPredWords: continue # don't include non-predictive words

        allWords.add(word)

        # tally the number of times a word is used in each bow
        if stars == 1:
          try:
            BOW[1][word] += 1 
          except:
            BOW[1][word] = 1 
        elif stars == 2:
          try:
            BOW[2][word] += 1
          except:
            BOW[2][word] = 1
        elif stars == 3:
          try:
            BOW[3][word] += 1
          except:
            BOW[3][word] = 1
       
        review['words'].append(word)
      reviews.append(review)

  # determine Bayesian probabiliy for each word in each bow
  for word in BOW[1]:
    BOW[1][word] /= len(BOW[1])
  for word in BOW[2]:
    BOW[2][word] /= len(BOW[2])
  for word in BOW[3]:
    BOW[3][word] /= len(BOW[3])

  # if words are in only one bow probability doesn't work, just give them a high value
  for word in allWords:
    if (word in BOW[1]) and (word not in BOW[2]) and (word not in BOW[3]):
      BOW[1][word] = 5
    if (word in BOW[2]) and (word not in BOW[1]) and (word not in BOW[3]):
      BOW[2][word] = 5
    if (word in BOW[3]) and (word not in BOW[1]) and (word not in BOW[2]):
      BOW[3][word] = 5

  # calculate each review's total weight for each BOW
  print('stars pred  bow1 weight  bow2 weight  bow3 weigt     words used')

  for review in reviews:
    oneStarScore   = 0
    twoStarScore   = 0
    threeStarScore = 0
  
    for word in review['words']:
      if word in BOW[1]:
        oneStarScore += BOW[1][word]
  
      if word in BOW[2]: 
        twoStarScore += BOW[2][word]

      if word in BOW[3]:
        threeStarScore += BOW[3][word]

    review['bowStars'] = (oneStarScore,twoStarScore,threeStarScore)

    # determine the BOW with the most weight
    pred = max(review['bowStars'])
    review['predStars'] = review['bowStars'].index(pred) + 1

    print('  ' + str(review['stars']) + '     ' +  str(review['predStars']) + "     {:.4f}".format(review['bowStars'][0]) + "       {:.4f}".format(review['bowStars'][1]) + "       {:.4f}".format(review['bowStars'][2]) + '     ' + str(review['words']))

  for review in reviews:
    # now check winner against actual stars given, print words and weights if no match
    if review['stars'] != review['predStars']:
      print()
      print('misclassified reviews:')
      print()
      print(review['text'])
      print()
      print('stars pred  bow1 weight  bow2 weight  bow3 weigt     words used')
      print('  ' + str(review['stars']) + '     ' +  str(review['predStars']) + "     {:.4f}".format(review['bowStars'][0]) + "       {:.4f}".format(review['bowStars'][1]) + "       {:.4f}".format(review['bowStars'][2]) + '     ' + str(review['words']))
      print()

      for word in review['words']:
        print(' ',word)
        if word in BOW[1]:
          print('     *',BOW[1][word])
        if word in BOW[2]:
          print('    **',BOW[2][word])
        if word in BOW[3]:
          print('   ***',BOW[3][word])
        print()

  # print results
  correct = 0
  wrong = 0
  for review in reviews:
    if review['stars'] == review['predStars']:
      correct += 1
    else:
      wrong += 1

  accuracy = (correct-wrong)/(correct+wrong)*100

  print('correct:',correct, '\nwrong:',wrong)
  print('accuracy: ' + "%.2f" % accuracy)
  print()

  # save the fit model
  pickle.dump(BOW, open("bayesModel.pkl","wb"))

def pred(infile):
  reviews = []
  allWords = set()

  BOW = pickle.load(open( "bayesModel.pkl","rb"))

  with open(infile) as inf:
    for count, line in enumerate(inf):
      line = line.strip()
      if not line: continue   # skip blank lines
      if count == 0: continue # skip the header

      review = dict(text=line, words=[], predStars=0, bowStars=(0,0,0))

      for word in line.split(' '):
      
        # clean the word
        word = word.strip()
        word = word.lower()
        word = re.sub(r'[^\w\s]', '', word)

        if word in stopWords: continue    # don't include stopwords
        if word in nonPredWords: continue # don't include non-predictive words

        review['words'].append(word)
      reviews.append(review)

  # calculate each review's total weight for each BOW
  for review in reviews:
    oneStarScore   = 0
    twoStarScore   = 0
    threeStarScore = 0

    for word in review['words']:
      if word in BOW[1]:
        oneStarScore += BOW[1][word]
  
      if word in BOW[2]: 
        twoStarScore += BOW[2][word]

      if word in BOW[3]:
        threeStarScore += BOW[3][word]

    review['bowStars'] = (oneStarScore,twoStarScore,threeStarScore)

    # determine the BOW with the most weight
    pred = max(review['bowStars'])
    review['predStars'] = review['bowStars'].index(pred) + 1
  
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


