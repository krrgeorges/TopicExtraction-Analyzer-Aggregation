import requests
from bs4 import BeautifulSoup as bs
import json
import time
import re
import random
import psycopg2
import pyperclip
from nltk.tag.stanford import StanfordNERTagger,StanfordPOSTagger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import OrderedDict




class ReviewsTopicsAggregator:
	USER_AGENTS = [
	    ('Mozilla/5.0 (X11; Linux x86_64) '
	     'AppleWebKit/537.36 (KHTML, like Gecko) '
	     'Chrome/57.0.2987.110 '
	     'Safari/537.36'),  # chrome
	    ('Mozilla/5.0 (X11; Linux x86_64) '
	     'AppleWebKit/537.36 (KHTML, like Gecko) '
	     'Chrome/61.0.3163.79 '
	     'Safari/537.36'),  # chrome
	    ('Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:55.0) '
	     'Gecko/20100101 '
	     'Firefox/55.0'),  # firefox
	    ('Mozilla/5.0 (X11; Linux x86_64) '
	     'AppleWebKit/537.36 (KHTML, like Gecko) '
	     'Chrome/61.0.3163.91 '
	     'Safari/537.36'),  # chrome
	    ('Mozilla/5.0 (X11; Linux x86_64) '
	     'AppleWebKit/537.36 (KHTML, like Gecko) '
	     'Chrome/62.0.3202.89 '
	     'Safari/537.36'),  # chrome
	    ('Mozilla/5.0 (X11; Linux x86_64) '
	     'AppleWebKit/537.36 (KHTML, like Gecko) '
	     'Chrome/63.0.3239.108 '
	     'Safari/537.36'),  # chrome
	]

	def __init__(self,product_link,ecommerce):
		'''
			Initialization of the Aggregator
			product_link : link to the product
			ecommerce: ecommerce site product link is referenced from. AMAZON => www.amazon.in, WALMART => www.walmart.in

		'''
		self.product_link = product_link
		self.ecommerce = ecommerce
		self.reviews = []
		self.conn = psycopg2.connect("dbname='dwdb' user='postgres' host='localhost' password='root'")
		self.cursor = self.conn.cursor()
		self.review_titles = []
		self.symbols_dict = "!-@#$%^&*(){}|:\"<>?[]\\;,./"

	def aggregateReviewsForAmazon(self):
		'''
			scraps amazon product link for reviews
		'''
		if "https://" in self.product_link:
			amazon_product_ref = url.split("/")[3]
			amazon_product_id = url.split("/")[5]
		else:
			amazon_product_ref = url.split("/")[1]
			amazon_product_id = url.split("/")[3]

		review_page_url = "https://www.amazon.in/"+amazon_product_ref+"/product-reviews/"+amazon_product_id+"/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber="
		page_no = 1;
		while True:
			try:
				soup = bs(requests.get(review_page_url+str(page_no),headers={ 'User-Agent': self.USER_AGENTS[random.randrange(len(self.USER_AGENTS))][0] }).content,"html.parser")
			except:
				with open("C://Users/Rojit/Desktop/reviews_test.json","r",encoding="utf-8") as f:
					self.reviews = json.loads(f.read())
				return
			review_titles = soup.find_all(lambda tag:tag.name=="a" and tag.get("data-hook")!=None and "review-title" in tag.get("data-hook"))
			review_stars = soup.find_all(lambda tag:tag.name=="i" and tag.get("data-hook")!=None and "review-star-rating" in tag.get("data-hook"))
			if len(review_stars) == 12:
				review_stars = review_stars[2::]
			review_bodies = soup.find_all(lambda tag:tag.name=="span" and tag.get("data-hook")!=None and "review-body" in tag.get("data-hook"))
			review_dates = soup.find_all(lambda tag:tag.name=="span" and tag.get("data-hook")!=None and "review-date" in tag.get("data-hook"))
			print(len(review_titles),len(review_stars),len(review_bodies),len(review_dates))
			for i in range(len(review_titles)):
				review = {}
				review["title"] = review_titles[i].text.strip()
				if review["title"] not in self.review_titles:
					review["stars"] = float(review_stars[i].text.strip().split(" ")[0])
					review["body"] = review_bodies[i].text.strip()
					review["date"] = " ".join(review_dates[i].text.strip().split(" ")[-3::])
					self.reviews.append(review)
					self.review_titles.append(review["title"])
			if len(self.reviews) > 200:
				break
			print(len(self.reviews))
			page_no += 1
		with open("C://Users/Rojit/Desktop/reviews_test.json","w+",encoding="utf-8") as f:
			f.write(json.dumps(self.reviews))

	def sentence_tokenize(self,review_body):
		'''
			gets all sentences except for questions.
		'''
		sentences = []
		delimiters = [".","!","?"]
		dot_sentences = re.split(r"\.|!",review_body)
		for ds in dot_sentences:
			temp_ds = ds
			if "?" in temp_ds:
				temp_ds = temp_ds.split("?")[-1]
			if len(temp_ds.strip()) > 0:
				sentences.append(temp_ds.strip())
		return sentences



	def is_english(self,sentence):
		'''
			checks if sentence is English or some other language.
		'''
		words = sentence.split(" ")
		limit = 0
		found = 0
		for r in words:
			if r in self.symbols_dict or len(r) == 0:
				found += 1
				continue
			self.cursor.execute("SELECT word from words_meanings where word=%(word)s",{"word":r})
			res = self.cursor.fetchall()
			if len(res) == 1:
				found += 1
		if found >= len(words)//2:
			return True

		return False


	def ngram_collector(self,sentence,n,two_steps=False,n2=0):
		'''
			collects ngram tuples for sentence
			n - number of words in single collection
		'''
		grams = []
		n2_grams = []
		pos_words = self.determine_sentpos_by_nltk(sentence)
		words = sentence.split(" ")
		if len(words) <= n:
			return [words]
		else:
			for i in range(len(words)):
				gram,n2_gram = [],[]
				gram_b,n2_gram_b = [],[]
				try:
					gram = pos_words[i:i+n]
					gram_b = words[i:i+n]
				except IndexError:
					break
				if two_steps == True:
					if n2 < n:
						n2_gram = gram[0:n2]
						n2_gram_b = gram_b[0:n2]
					if n2 > n:
						try:
							n2_gram = gram + pos_words[i+n:(i+n)+(n2-n)]
							n2_gram_b = gram_b + words[i+n:(i+n)+(n2-n)]
						except IndexError:
							lol = 1
				symbol_check,symbol_check_n2 = True,True
				for s in self.symbols_dict:
					if s in gram_b:
						symbol_check = False
					if s in n2_gram_b:
						symbol_check_n2 = False
				if len(gram) == n and symbol_check == True:
					grams.append(gram)
				if len(n2_gram) == n2 and symbol_check_n2 == True:
					n2_grams.append(n2_gram)


		if two_steps == True:
			return [grams,n2_grams]
		return grams

	def handle_symbols(self,review_body):
		'''
			does necessary symbols handling in review body for proper ngram collection and pos determination
		'''
		for s in self.symbols_dict:
			review_body = review_body.replace(s," "+s+" ")
		return review_body

	def determine_wordpos_by_db(self,word):
		'''
			gets pos for word from bs
		'''
		self.cursor.execute("SELECT types FROM words_meanings WHERE word=%(word)s",{"word":word})
		res = self.cursor.fetchall()
		if len(res) == 0:
			return []
		return res[0][0]


	def get_word_meaning_by_db(self,word):
		'''
			gets pos for word from bs
		'''
		self.cursor.execute("SELECT meanings FROM words_meanings WHERE word=%(word)s",{"word":word})
		res = self.cursor.fetchall()
		if len(res) == 0:
			return []
		return res[0][0]

	def determine_sentpos_by_nltk(self,sentence):
		'''
			get pos collection for sentence from nltk
		'''
		pos_model_file = "C://python34/ProjectDragonWolf/nlp_res/stanford_pos/models/english-bidirectional-distsim.tagger"
		pos_jar_file = "C://python34/ProjectDragonWolf/nlp_res/stanford_pos/stanford-postagger.jar"
		pos = StanfordPOSTagger(model_filename=pos_model_file,path_to_jar=pos_jar_file)
		return pos.tag(sentence.split(" "))


	def is_english(self,pos_sentence):
		is_english = True
		n_fw = 0
		for p in pos_sentence:
			if p[1] == 'FW':
				n_fw += 1
		if n_fw >= len(pos_sentence)//2:
			is_english = False

		return is_english



	def remove_stopwords(self,pos_sentence,tps):
		stopwords = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]

		for p in pos_sentence:
			if p[0] in stopwords or len(p[0]) <= 3:
				pos_sentence.remove(p)
				try:
					tps.remove(p)
				except:
					continue

		return pos_sentence,tps

	def combine_consequent_same_pos_words(self,consider_tags,noun_consideration,pos_sentence):
		tps = []
		i = 0
		is_valid =  False

		while True:
			try:
				if len(pos_sentence[i][0]) < 3:
					i += 1
					continue
				if pos_sentence[i][1] in consider_tags:
					is_valid = True
			except IndexError:
				break
			try:				
				a = pos_sentence[i+1]
			except IndexError:
				tps.append((pos_sentence[i][0],pos_sentence[i][1]))
				break
			if pos_sentence[i][1] == pos_sentence[i+1][1]:
				tps.append((pos_sentence[i][0]+" "+pos_sentence[i+1][0],pos_sentence[i][1]))
				i += 2
			else:
				tps.append((pos_sentence[i][0],pos_sentence[i][1]))
				i += 1

		return is_valid,pos_sentence,tps


	def middle_out_assigner(self,tps,consider_word,window_limit,consider_tags,noun_consideration,i):
		ln,rn = -1,-1
		iln,irn = i-1,i+1
		lw,rw = "",""

		for j in range(0,window_limit):
			try:
				if tps[iln][1] in noun_consideration:
					ln = iln
					lw = tps[iln][0]
					break
				iln -=1
			except IndexError:
				break

		for j in range(0,window_limit):
			try:
				if tps[irn][1] in noun_consideration:
					rn = irn
					rw = tps[irn][0]
					break
				irn +=1
			except IndexError:
				break

		return ln,rn,lw,rw


	def sentiment_determiner(self,analyzer,sentence,stars):
		'''
			obtains polarity score for sentence and determines primary sentiment.
		'''
		sentiment_scores = analyzer.polarity_scores(sentence)
		pos = sentiment_scores["pos"]
		neu = sentiment_scores["neu"]
		neg = sentiment_scores["neg"]
		compound = sentiment_scores["compound"]

		stars_sentiment = 0

		if stars >= 4.0:
			stars_sentiment = 1

		msentiment = None

		if neg == 1.0:
			msentiment = 0
		elif pos == 1.0:
			msentiment = 1
		elif neu == 1.0:
			if stars_sentiment == 0:
				msentiment = 0
			else:
				msentiment = 1

		if msentiment == None:
			if pos == 0 and neg != 0:
				msentiment = 0
			if pos != 0 and neg == 0:
				msentiment = 1
			if neu == 0 and pos > neg:
				msentiment = 1
			if neu == 0 and neg > pos:
				msentiment = 0

			if msentiment == None:
				if pos > neu and pos > neg:
					msentiment = 1
				if neu > pos and neu > neg:
					msentiment = stars_sentiment
				if neg > neu and neg > pos:
					msentiment = 0



		if msentiment == None:
			msentiment = stars_sentiment
		# else:
		# 	if stars_sentiment > msentiment:
		# 		msentiment = stars_sentiment

		return msentiment



		



	def aggregate_topics(self):
		'''
			main function. aggregates all topics by gathering reviews, obtaining topics mentioned in the review with sentiment. and using custom words db to group similar topics.
		'''
		topics = []
		if self.ecommerce == "AMAZON":
			self.aggregateReviewsForAmazon()

		sentiment_analyzer = SentimentIntensityAnalyzer()

		for review in self.reviews:
			body = self.handle_symbols(review["body"].replace("\n"," ").lower().strip())
			sentences = self.sentence_tokenize(body)
			for s in sentences:
				# pos tagging
				try:
					pos_sentence = self.determine_sentpos_by_nltk(s)
				except:
					continue
				
				#check if english
				is_english = self.is_english(pos_sentence)
				

				if is_english == True:
					# combine consequent simiar pos words
					consider_tags = ["VBN","JJ","JJS"]
					noun_consideration = ["NN","NNS"]
					is_valid,pos_sentence,tps = self.combine_consequent_same_pos_words(consider_tags,noun_consideration,pos_sentence)
					
					pos_sentence,tps = self.remove_stopwords(pos_sentence,tps)

					# left and right check 
					if is_valid == True:
						sentiment_scores = None

						for i in range (len(tps)):
							if tps[i][1] in consider_tags and len(tps[i][0]) >= 4:
								consider_word = tps[i][0]
								window_limit = 4

								ln,rn,lw,rw = self.middle_out_assigner(tps,consider_word,window_limit,consider_tags,noun_consideration,i)

								phrase,mw,m2 = "","",""

								if ln == -1 and rn > -1:
									phrase = tps[i:rn+1]
									mw = rw
								elif ln > -1 and rn == -1:
									phrase = tps[ln:i+1]
									mw = lw
								elif ln > -1 and rn > -1:
									if rn-i < i-ln:
										phrase = tps[i:rn+1]
										mw = rw
									elif i-ln < rn-i:
										phrase = tps[ln:i+1]
										mw = lw
									else:
										phrase = tps[ln:rn+1]
										mw = lw
										mw2 = rw



								if len(phrase) <= 3 and mw != "":
									consider_word_pos = self.determine_wordpos_by_db(consider_word)
									if "adverb" not in consider_word_pos and "pronoun" not in consider_word_pos and 'adjective' in consider_word_pos:
										if sentiment_scores == None:
											sentiment_scores = sentiment_analyzer.polarity_scores(s)
										stars = review["stars"]
										print(stars)
										m_sentiment_score = self.sentiment_determiner(sentiment_analyzer,sentiment_scores,stars)
										print(sentiment_scores)
										print(str(phrase).encode(),consider_word.encode(),mw.encode())
										topics.append((consider_word,mw,m_sentiment_score))
										print(s.encode())
										


		#cluster

		topic_names = []
		topic_clusters = {}
		topic_meanings = {}

		for t in topics:
			tn = t[1]
			an = t[0]
			s_s = t[2]
			if tn not in topic_clusters:
				topic_clusters[tn] = {an:[1,[s_s]]}
				if tn not in topic_names:
					topic_names.append(tn)
			else:
				consider_list = topic_clusters[tn]
				if an not in consider_list:
					consider_list[an] = [1,[s_s]]
				else:
					consider_list[an][1].append(s_s) 
					consider_list[an][0] = consider_list[an][0]+1
				topic_clusters[tn] = consider_list
				if tn not in topic_names:
					topic_names.append(tn)

			if tn not in topic_meanings:
				topic_meanings[tn] = self.get_word_meaning_by_db(tn)

		#combine similar topics



		for i in range(len(topic_names)):
			topic = topic_names[i]
			try:
				considers = topic_clusters[topic]
			except KeyError:
				continue
			meanings = topic_meanings[topic]
			for j in range(i+1,len(topic_names)):
				try:
					ntopic = topic_names[j]
					if " " in ntopic:
						ntopic = topic_names[j].split(" ")[-1]
				except KeyError:
					continue
				try:
					nconsiders = topic_clusters[topic_names[j]]
				except KeyError:
					continue
				nmeanings = self.get_word_meaning_by_db(ntopic)
				for m in nmeanings:
					if m in meanings:
						for c in nconsiders:
							if c not in considers:
								considers[c] = nconsiders[c]
							else:
								for k in nconsiders[c][1]:
									considers[c][1].append(k)
								considers[c][0] = considers[c][0] + nconsiders[c][0]
						topic_clusters[topic] = considers
						del topic_clusters[topic_names[j]]
						del topic_meanings[topic_names[j]]
						break


		stns = []


		judge = 0
		#get topics with adjectives more than judge
		del_t_keys = []
		topic_names = []
		for t in topic_clusters:
			topic_names.append(t)
			found_biggie = False
			mw_dict = topic_clusters[t]
			del_keys = []

			for m in mw_dict:
				if mw_dict[m][0] > judge:
					found_biggie = True
				else:
					del_keys.append(m)

			for d in del_keys:
				del mw_dict[d]
			if found_biggie == False or len(mw_dict) < 2:
				del_t_keys.append(t)
			else:
				topic_clusters[t] = mw_dict


		for t in del_t_keys:
			del topic_clusters[t]

		#sorting

		stns = OrderedDict()

		tns = [k for k in topic_clusters]

		for i in range(len(tns)):
			t = tns[i]
			scns = OrderedDict()
			consider_names = []
			try:
				consider_words = topic_clusters[t]
			except KeyError:
				continue
			for c in consider_words:
				consider_names.append(c)
			for j in range(0,len(consider_names)):
				c1 = consider_names[j]
				c1_consider_count = consider_words[c1][0]
				for k in range(j+1,len(consider_names)):
					c2 = consider_names[k]
					c2_consider_count = consider_words[c2][0]
					if c1_consider_count > c2_consider_count:
						a = consider_names[j]
						consider_names[j] = consider_names[k]
						consider_names[k] = a
			for c in consider_names:
				scns[c] = consider_words[c]
			topic_clusters[t] = scns

		for i in range(len(tns)):
			for j in range(i+1,len(tns)):
				try:
					if len(topic_clusters[tns[i]]) > len(topic_clusters[tns[j]]):
						a = tns[i]
						tns[i] = tns[j]
						tns[j] = a
				except KeyError:
					continue

		for t in tns:
			try:
				stns[t] = topic_clusters[t]
			except:
				continue



		print(stns)



url = "https://www.amazon.in/Tata-Salt-1kg/dp/B01HBF5WBI/ref=pd_rhf_gw_s_pd_crcd_0_1/261-5422482-6234051?_encoding=UTF8&pd_rd_i=B01HBF5WBI&pd_rd_r=a3a551e0-9e3a-4447-a701-26b1ccc0a9be&pd_rd_w=Gxf1K&pd_rd_wg=fwc1J&pf_rd_p=decc33e5-9bac-4f5a-9e32-c8b613fff04e&pf_rd_r=TPRN1YG60XS7XH6MXGJQ&psc=1&refRID=TPRN1YG60XS7XH6MXGJQ"
ReviewsTopicsAggregator(url,"AMAZON").aggregate_topics()


