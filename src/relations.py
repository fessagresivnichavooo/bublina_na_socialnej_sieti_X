import copy
import networkx as nx
import matplotlib.pyplot as plt
import test_twitter_scrapper_from_json
import AIAnalysis
import json
from fuzzywuzzy import process, fuzz
import math
from collections import Counter

### follower/followed

### profily repostov a reakcii, ideal ked to je aj sledovane tou osobou (moze to byt aj repost nahodne vyskoceneho tweetu)
#   v takomto pripade ulozi ten repost k profilu

### followovane ktore nefollowuju nikoho v skupine naspat


### na zaver moze byt vyhodnocovanie miery interakcii (like, repost)



### decentralised => zoberie pocet profilov, spravi analyzu na kazdy, definuje pripadne hrany a spolocne vlastnosti, hrana nie je interakcia ale vlastnost spolocna


### FAZY

#  1. inicializacia => prejde profily a vlozi do grafu, vytvori strukturu pre tweety
#  2. analyza interakcii => edges, like comment medzi clenmi bubliny, ciel je zistit weights
### vytvorit mnozinu tweetov na zaklade ich status id
#  2.9. analyza profilov outside bubble -> celebrita, fanpage, firma, zabava/vzdelanie, ostatne
#  3. analyza tweetov => topics, semantics
#  4. analyza dat bubliny => interpretacia ziskanych dat



### vlastne nazory v tweetoch, reakcie na outside of bubble, sledovanie outside of bubble (podla poctu sledovani, ak je viac tak ai nech vyhodnoti kto to je a co robi,
### ak nevie tak skusi podla popisu profilu, ak je pocet followerov nizsi, vyskusa ale ak nie tak nie)


## check followy, reakcie

## jednostranny follow niekoho z bubliny, obojstranny follow, follow + reakcia jednostranny, obojstranna



SCRAPPER = test_twitter_scrapper_from_json.TwitterScrapper()
PROFILE_AI_ANALYSER = AIAnalysis.GPT4o()
TWEET_AI_ANALYSER = AIAnalysis.GPT4o()


ALL_TWEETS = {}
OUTSIDE_BUBBLE_PROFILES_ANALYSED = {}
THRESHOLD = 2000
OTHER_TOPICS = ["Finance", "Entertainment", "Technology", "Education"]


class Node:
    def __init__(self, username):
        self.edges = []
        self.profile = Profile(username)

class Edge:
    def __init__(self, weight, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.directions = {
            node1 : node2,
            node2 : node1
        }
        ### 1 reaguje na 2    1->2
        self.weight = {"follows": None, "reactions": {"1->2":[], "2->1":[]}, "mentions": {"1->2": 0, "2->1": 0}}
        if weight in ["friends", "->", "<-", "X"]:
            self.weight["follows"] = weight
        else:
            raise ValueError("Incorrect value for weight")

            
        ### follows: friends, 1->2, 2->1
        ### reactions: ...
        #self.interactions = {likes, comments, reposts}

    def get_second_node(self, node):
        return self.directions[node]

    ### 1 reaguje na 2
    def direction(self, node1: Node, node2: Node):
        if node1 == self.node1 and node2 == self.node2:
            return "1->2"
        elif node2 == self.node1 and node1 == self.node2:
            return "2->1"
        else:
            raise ValueError("invalid node parameter")

    def __str__(self):
        return f"{self.node1.profile.username}; {self.node2.profile.username} | {self.weight}"


class Summary:
    def __init__(self, username, expression, interaction, interest, time_interval=None):
        self.username = username
        self.expression_sum = expression
        self.interaction_sum = interaction
        self.interest_sum = interest
        print(self.username, self.expression_sum, self.interaction_sum, self.interest_sum)
        self.overall = self.overall_sum()

##    def interpret_sentiment_list(self, sentiments, pos, neu, neg):
##        mapping = {'positive': pos, 'neutral': neu, 'negative': neg}
##        scores = [mapping[s] for s in sentiments]
##        raw_score = sum(scores) / len(scores)
##        confidence = min(1.0, math.log2(len(scores) + 1) / 3)
##        return raw_score * confidence
        import math

    def interpret_sentiment_list(self, sentiments, pos, neu, neg):
        mapping = {'positive': pos, 'neutral': neu, 'negative': neg}
        scores = [mapping[s] for s in sentiments]
        
        if not scores:
            return 0  # Avoid division by zero

        raw_score = sum(scores) / len(scores)

        n = len(scores)
        confidence = n / (n + 3.5)  # Slower start, then grows faster

        return raw_score * confidence


    def overall_sum(self):
        ### SPORT
        ''' poradie podla weight h->l: expression >= interaction >(o trochu) interest '''
        self.sports = {}
        self.clubs = {}
        self.athletes = {}
        self.genres = {}
        self.artists = {}

        for sport, data in self.expression_sum["sports"].items():
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.5)
            mentions = len(data["sentiments"]) + data["c/p tweets"]
            self.sports[sport] = {"sentiment": [sentiment], "mentions": [mentions]}

        for sport, data in self.interaction_sum["sports"].items():
            sentiment = self.interpret_sentiment_list(self.data["sentiments"], 1.1, 0.4, -0.5)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + data["c/p tweets"] + len(data["reaction sentiments"])
            temp = process.extractOne(sport, self.sports.keys()) or [0,0]
            if sport not in self.sports and temp[1] < 85:
                self.sports[sport.lower()] = {"sentiment": [sentiment], "mentions": [mentions]}
            else:
                self.sports[temp[0]]["sentiment"].append(sentiment)
                self.sports[temp[0]]["sentiment"].append(reaction_sentiment)
                self.sports[temp[0]]["mentions"].append(mentions)

        if self.interest_sum["sport"]:
            for sport in self.interest_sum["sport"]:
                sentiment = self.interpret_sentiment_list(["positive"]*sport["counter"], 0.6, 0, 0)
                if sport["sport"].lower() not in self.sports:
                    self.sports[sport["sport"].lower()] = {"sentiment": [sentiment], "mentions": [sport["counter"]]}
                else:
                    self.sports[sport["sport"].lower()]["sentiment"].append(sentiment)
                    self.sports[sport["sport"].lower()]["mentions"].append(sport["counter"])
                

        for club, data in self.expression_sum["clubs"].items():
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.7)
            mentions = len(data["sentiments"])
            self.clubs[club] = {"sentiment": [sentiment], "mentions": [mentions]}

        for club, data in self.interaction_sum["clubs"].items():
            sentiment = self.interpret_sentiment_list(self.data["sentiments"], 1.1, 0.4, -0.7)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + len(data["reaction sentiments"])
            temp = process.extractOne(club, self.clubs.keys()) or [0,0]

            sport = self.expression_sum.get("clubs", {}).get(temp[0], None)["sports"] or self.interaction_sum.get("clubs", {}).get(temp[0], None)["sports"] or self.interest_sum.get("clubs", {}).get(temp[0], None)["sports"]
            if sport is None:
                print("NO SPORT")
                self.clubs[club] = {"sentiment": [sentiment], "mentions": [mentions]}
            else:
                sport = Counter(sport).most_common(1)[0][0]
                
            if club not in self.clubs and temp[1] < 85:
                self.clubs[club] = {"sentiment": [sentiment], "mentions": [mentions]}
                
            elif self.clubs[temp[0]] and fuzz.ratio(sport, Counter(data["sports"]).most_common(1)[0][0]) > 87:
                self.clubs[temp[0]]["sentiment"].append(sentiment)
                self.clubs[temp[0]]["sentiment"].append(reaction_sentiment)
                self.clubs[temp[0]]["mentions"].append(mentions)
                
            else:
                self.clubs[club] = {"sentiment": [sentiment], "mentions": [mentions]}
                

        if self.interest_sum["sport"]:
            for sport in self.interest_sum["sport"]:
                if sport["countries"]:
                    for country in sport["countries"]:
                        if country["clubs"]:
                            for club in set(country["clubs"]):
                                if self.clubs:
                                    temp = process.extractOne(club.lower(), self.clubs.keys())
                                    #print(temp, club)
                                    if temp[1] > 85:
                                        mentions = country["clubs"].count(club)
                                        sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                        self.clubs[temp[0].lower()]["sentiment"].append(sentiment)
                                        self.clubs[temp[0].lower()]["mentions"].append(mentions)
                                    else:
                                        mentions = country["clubs"].count(club)
                                        sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                        self.clubs[club.lower()] = {"sentiment": [sentiment], "mentions": [mentions]}
                                        
                                elif club.lower() not in self.clubs:
                                    mentions = country["clubs"].count(club)
                                    sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                    self.clubs[club.lower()] = {"sentiment": [sentiment], "mentions": [mentions]}

            
        for player, data in self.expression_sum["players"].items():
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.7)
            mentions = len(data["sentiments"])
            self.athletes[player] = {"sentiment": [sentiment], "mentions": [mentions]}

        for player, data in self.interaction_sum["players"].items():
            sentiment = self.interpret_sentiment_list(self.data["sentiments"], 1.1, 0.4, -0.7)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + len(data["reaction sentiments"])
            if player not in self.players:
                self.athletes[player] = {"sentiment": [sentiment], "mentions": [mentions]}
            else:
                self.athletes[player]["sentiment"].append(sentiment)
                self.athletes[player]["sentiment"].append(reaction_sentiment)
                self.athletes[player]["mentions"].append(mentions)

        if self.interest_sum["sport"]:
            for sport in self.interest_sum["sport"]:
                if sport["countries"]:
                    for country in sport["countries"]:
                        if country["athletes"]:
                            for athlete in set(country["athletes"]):
                                if self.athletes:
                                    temp = process.extractOne(athlete.lower(), self.athletes.keys())
                                    if temp[1] > 85:
                                        mentions = country["athletes"].count(temp[0])
                                        sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                        self.athletes[temp[0].lower()]["sentiment"].append(sentiment)
                                        self.athletes[temp[0].lower()]["mentions"].append(mentions)
                                    else:
                                        mentions = country["athletes"].count(athlete)
                                        sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                        self.athletes[athlete.lower()] = {"sentiment": [sentiment], "mentions": [mentions]}
                                        
                                elif athlete.lower() not in self.athletes:
                                    mentions = country["athletes"].count(athlete)
                                    sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                    self.athletes[athlete] = {"sentiment": [sentiment], "mentions": [mentions]}

        for genre, data in self.expression_sum["genres"].items():
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.5)
            mentions = len(data["sentiments"]) + data["artist tweets"]
            self.genres[genre] = {"sentiment": [sentiment], "mentions": [mentions]}

        for genre, data in self.interaction_sum["genres"].items():
            sentiment = self.interpret_sentiment_list(self.data["sentiments"], 1.1, 0.4, -0.5)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + data["artist tweets"] + len(data["reaction sentiments"])
            temp = process.extractOne(genre, self.genres.keys()) or [0,0]
            if genre not in self.genres and temp[1] < 85:
                self.genres[genre.lower()] = {"sentiment": [sentiment], "mentions": [mentions]}
            else:
                self.genres[temp[0]]["sentiment"].append(sentiment)
                self.genres[temp[0]]["sentiment"].append(reaction_sentiment)
                self.genres[temp[0]]["mentions"].append(mentions)

        if self.interest_sum["music"]:
            for genre in self.interest_sum["music"]:
                sentiment = self.interpret_sentiment_list(["positive"]*genre["counter"], 0.6, 0, 0)
                if genre["genre"].lower() not in self.genres:
                    self.genres[genre["genre"].lower()] = {"sentiment": [sentiment], "mentions": [genre["counter"]]}
                else:
                    self.genres[genre["genre"].lower()]["sentiment"].append(sentiment)
                    self.genres[genre["genre"].lower()]["mentions"].append(genre["counter"])



        for artist, data in self.expression_sum["artists"].items():
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.7)
            mentions = len(data["sentiments"])
            self.artists[artist] = {"sentiment": [sentiment], "mentions": [mentions]}

        for artist, data in self.interaction_sum["players"].items():
            sentiment = self.interpret_sentiment_list(self.data["sentiments"], 1.1, 0.4, -0.7)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + len(data["reaction sentiments"])
            if artist not in self.artists:
                self.artists[artist] = {"sentiment": [sentiment], "mentions": [mentions]}
            else:
                self.artists[artist]["sentiment"].append(sentiment)
                self.artists[artist]["sentiment"].append(reaction_sentiment)
                self.artists[artist]["mentions"].append(mentions)

        if self.interest_sum["music"]:
            for genre in self.interest_sum["music"]:
                if genre["artists"]:
                    for artist in set(genre["artists"]):
                        if self.artists:
                            temp = process.extractOne(artist.lower(), self.artists.keys())
                            if temp[1] > 85:
                                mentions = genre["artists"].count(temp[0])
                                sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                self.artists[temp[0].lower()]["sentiment"].append(sentiment)
                                self.artists[temp[0].lower()]["mentions"].append(mentions)
                            else:
                                mentions = genre["artists"].count(artist)
                                sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                self.artists[artist.lower()] = {"sentiment": [sentiment], "mentions": [mentions]}
                                
                        elif artist.lower() not in self.artists:
                            mentions = genre["artists"].count(artist)
                            sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                            self.artists[artist] = {"sentiment": [sentiment], "mentions": [mentions]}

                   
        



        
        with open("test.json", 'w', encoding="utf-8") as file:
            json.dump([self.sports, self.clubs, self.athletes, self.genres, self.artists], file, indent=4)

        
            
        
        
'''
najviac reaguje na temu


jazyky pomer
oblasti zaujmu poradie

sporty - fav , nefav    ku kazdemu top krajiny, max 3
kluby - fav , nefav
hraci - fav , nefav     -//- ak prevazuje krajina nejaka, spomenut
zanre -//-              -//- top krajiny ak nejaka vyrazne
hudobnici -//-          ak nejaka prevazuje
politika
    prevazujuca ideologia
    krajiny zaujmu


SPORT:
    mentions: pocet tweetov + counter following
    sentiment: iba sport ako taky
    
'''
    


class Profile:
    def __init__(self, username):
        self.username = username
        self.bio, self.followers, self.following, self.profession, self.location = SCRAPPER.scrape_profile(username)
        self.friends = list(set(copy.deepcopy(self.following)) & set(copy.deepcopy(self.followers)))

        self.tweets = []
        self.reposts = []
        self.comments = []
        self.quotes = []
        self.all_mentions = {}
        
        for status_id, username, text, type, source_tweet, source_username, hashtags, mentions in SCRAPPER.get_tweets(username):
            #print(type, source_tweet)
##            if status_id == "1877127950253822340":
##                print(self.username, status_id, username, text, type, source_tweet, source_username, hashtags, mentions, "\na\na\na\na\na\na\na\na\na\na\na\n")
            tweet = Tweet(status_id, username, text, type, source_tweet, source_username, hashtags, mentions)
            ALL_TWEETS[status_id] = tweet
            if tweet.get_type() == "repost":
                self.reposts.append(tweet)
            elif tweet.get_type() == "comment":
                self.comments.append(tweet)
            elif tweet.get_type() == "tweet":
                self.tweets.append(tweet)
            elif tweet.get_type() == "quote":
                self.quotes.append(tweet)
            else:
                raise ValueError("Incorrect type value")
            for mention in mentions:
                if mention not in self.all_mentions:
                    self.all_mentions[mention] = 0
                self.all_mentions[mention] += 1
            

    def following_outside_bubble_big_profiles(self, following_outside_bubble_list, follower_count):
        return_val = []
        for screen_name, name, followers, description in SCRAPPER.get_following_extended(self.username):
            if followers > follower_count and screen_name in following_outside_bubble_list:
                return_val.append((screen_name, name, description))
        return return_val
        
    def __repr__(self):
        return (
            f"Profile(username={self.username!r}, profession={self.profession!r}, location={self.location!r})\n"
            f"Bio: {self.bio}\n"
            f"Followers: {len(self.followers)}, Following: {len(self.following)}, Friends: {len(self.friends)}\n"
            f"Tweets: {len(self.tweets)}, Reposts: {len(self.reposts)}, Comments: {len(self.comments)}, Quotes: {len(self.quotes)}\n"
        )

    def summary(self, time_interval=None, AI=False):
        ### vyskusat normalne
        ### vyskusat s AI

        ### rozdelit si sumar a analyzu na jednoducho vydedukovatelne a tazko
        ### napr. jednoducho = sleduje vela americkych novinarov -> zaujima sa a o americku politiku
##        print(self.username)
##        print(self.expression_sum())
##        print(self.interaction_sum())
##        print(self.interest_sum("profile_analysis.json"))
        Summary(self.username, self.expression_sum(), self.interaction_sum(), self.interest_sum("profile_analysis.json"))
##        print('\n\n\n\n')
        


    def expression_sum(self, time_interval=None):
        expression_summary = {
            "politics": {
                "type": [],
                "no type": []
            },
            "languages": [],
            "sports": {},   # sport: {sentiment when just itself: str, c/p number tweets}
            "clubs": {},    # club: {[sentiments], country, [sports]}
            "players": {},  # player: {[sentiments], country, sport}
            "genres": {},   # genre: {[countries], [countries_negative], [sentiments]}
            "artists": {}   # artist:{genre, country, [sentiments]}
        }
        
        for tweet in self.tweets+self.reposts:
            try:
                ############## OPRAVIT TENTO ERROR ######################
                type = tweet.content["type"]
            except TypeError:
                print("ERROR expression tweet.content[\"type\"]", tweet.text, ALL_TWEETS)
                continue
            if tweet.content["sport"]:
                clubs = tweet.content["sport"].get("clubs", [])
                players = tweet.content["sport"].get("players", [])
            else:
                clubs, players = [], []
            
            if type == "politics":
                expression_summary["politics"]["type"].append(tweet.content["politics"])
            else:
                expression_summary["politics"]["no type"].append(tweet.content["politics"])
            expression_summary["languages"].append(tweet.content["language"])

            if type == "sport":
                temp_club_player_sports = set()
                if clubs:
                    for club in clubs:   ### v pripade nerozoznania rovnakych klubov-> fuzzy wuzzy + porovnat krajiny
                        if club["club"] not in expression_summary["clubs"] and not process.extractOne(club["club"], list(expression_summary["clubs"].keys())):
                            expression_summary["clubs"][club["club"].lower()] = {"sentiments": [], "country": "", "sports": []}
                        elif process.extractOne(club["club"], list(expression_summary["clubs"].keys()))[1] > 85:
                            club["club"] = process.extractOne(club["club"], list(expression_summary["clubs"].keys()))[0]
                        else:
                            expression_summary["clubs"][club["club"].lower()] = {"sentiments": [], "country": "", "sports": []}
                                
                        expression_summary["clubs"][club["club"].lower()]["sentiments"].append(club["sentiment"].lower())
                        expression_summary["clubs"][club["club"].lower()]["country"] = club["country"].lower()
                        expression_summary["clubs"][club["club"].lower()]["sports"].append(club["sport"].lower())
                        temp_club_player_sports.add(club["sport"].lower())

                if players:
                    for player in players:   ### v pripade nerozoznania rovnakych klubov-> fuzzy wuzzy + porovnat krajiny
                        if player["player"] not in expression_summary["players"] and not process.extractOne(player["player"], list(expression_summary["players"].keys())):
                            expression_summary["players"][player["player"].lower()] = {"sentiments": [], "country": "", "sport": []}
                        elif process.extractOne(player["player"], list(expression_summary["players"].keys()))[1] > 85:
                            player["player"] = process.extractOne(player["player"], list(expression_summary["players"].keys()))[0]
                        else:
                            expression_summary["players"][player["player"].lower()] = {"sentiments": [], "country": "", "sport": []}
                        
                        expression_summary["players"][player["player"].lower()]["sentiments"].append(player["sentiment"].lower())
                        expression_summary["players"][player["player"].lower()]["country"] = player["country"].lower()
                        expression_summary["players"][player["player"].lower()]["sport"] = player["sport"].lower()
                        temp_club_player_sports.add(player["sport"].lower())
                
                for sport in tweet.content["sport"]["sports"]:
                    if sport["sport"].lower() not in expression_summary["sports"]:
                        expression_summary["sports"][sport["sport"].lower()] = {"sentiments": [], "c/p tweets": 0}  ### sentiments su iba pre tie sporty, ktore su spomenute individualne
                    if sport["sport"].lower() in temp_club_player_sports:
                        expression_summary["sports"][sport["sport"].lower()]["c/p tweets"] += 1
                    else:
                        expression_summary["sports"][sport["sport"].lower()]["sentiments"].append(sport["sentiment"].lower())

            if type == "music":
                temp = []
                genres = tweet.content["music"].get("genres", [])
                artists = tweet.content["music"].get("artists", [])

                if artists:
                    for artist in artists:
                        if artist["artist"] not in expression_summary["artists"]:
                            expression_summary["artists"][artist["artist"].lower()] = {"sentiments": [], "country": "", "genres": []}
                        expression_summary["artists"][artist["artist"].lower()]["sentiments"].append(artist["sentiment"].lower())
                        expression_summary["artists"][artist["artist"].lower()]["country"] = artist["country"].lower()
                        expression_summary["artists"][artist["artist"].lower()]["genres"].append(artist["genre"].lower())
                        temp.append(artist["genre"].lower())
                        
                if genres:
                    for genre in genres:
                        if genre["genre"].lower() not in expression_summary["genres"]:
                            expression_summary["genres"][genre["genre"].lower()] = {"sentiments": [], "artist tweets": 0}
                        expression_summary["genres"][genre["genre"].lower()]["artist tweets"] += temp.count(genre["genre"].lower())
                        expression_summary["genres"][genre["genre"].lower()]["sentiments"].append(genre["sentiment"].lower())
                        
        return expression_summary

    
    def interaction_sum(self, time_interval=None):
        reaction_translate = {
            "Disagreeing": {
                "neutral": "neutral",
                "positive": "negative",
                "negative": "positive",
                "none": "neutral"
            },
            "Agreeing": {
                "neutral": "neutral",
                "positive": "positive",
                "negative": "negative",
                "none": "neutral"
            },
            "Neutral": {  ### na neutral preto, lebo je to spomenute, ale nie v specifickom sentimente
                "neutral": "neutral",
                "positive": "neutral",
                "negative": "neutral",
                "none": "neutral"
            }
        }
        interaction_summary = {
            "politics": {
                "type": [],
                "type reaction": [],
                "no type": [],
                "no type reaction": []
            },
            "languages": [],
            "sports": {},   # ak je text same o sebe cisto o sporte (nie club/player), tak vezme to, + reakcia -> aj pri other, ak je to disagree tak reverse sentimentu predosleho tweetu ak agree tak copy

                            # sport: {sentiment when just itself: str, reaction sentiment (reversed if disagree)}
            "clubs": {},    # club: {[sentiments], country, [sports]}
            "players": {},  # player: {[sentiments], country, sport}
            "genres": {},   # genre: {[countries], [countries_negative], [sentiments]}
            "artists": {}   # artist:{genre, country, [sentiments]}
        }
        for reaction in self.quotes+self.comments:
            if self.username != reaction.username:
                continue
            try:
                ############## OPRAVIT TENTO ERROR ######################
                type = reaction.content["reaction"]["type"]
            except TypeError:
                print("\nERROR: type = reaction.content[\"reaction\"][\"type\"]  ", reaction.text, reaction.content,ALL_TWEETS)
                continue
            except KeyError:
                reaction.content["reaction"] = reaction.content.copy()
                
            if reaction.content["reaction"]["sport"]:
                clubs = reaction.content["reaction"]["sport"].get("clubs", [])
                players = reaction.content["reaction"]["sport"].get("players", [])
            else:
                clubs, players = [], []

            if reaction.content["reaction"]["music"]:
                genres = reaction.content["reaction"]["music"].get("genres", [])
                artists = reaction.content["reaction"]["music"].get("artists", [])
            else:
                genres, artists = [], []
            

            if not ALL_TWEETS.get(reaction.source_tweet, None):
                print("\nZMAZANY OG TWEET  ", reaction.source_tweet, "\n")
                continue
            
            source_tweet_content = ALL_TWEETS[reaction.source_tweet].content
            
            
            if "reaction" in source_tweet_content:
                source_tweet_content = source_tweet_content["reaction"]

                
            if type == "politics" or source_tweet_content["type"] == "sport":
                interaction_summary["politics"]["type"].append(reaction.content["reaction"]["politics"])
                #################### DOROBIT TUTO KED SA SFINALIZUJE FORMAT POLITIKY ####################
                '''expression_summary["politics"]["type reaction"].append(source_tweet_content["type"])'''
            else:
                interaction_summary["politics"]["no type"].append(reaction.content["reaction"]["politics"])
            interaction_summary["languages"].append(reaction.content["reaction"]["language"])

            
            if type == "sport" or source_tweet_content["type"] == "sport":
                temp_club_player_sports = set()
                if clubs:
                    for club in clubs:   ### v pripade nerozoznania rovnakych klubov-> fuzzy wuzzy + porovnat krajiny
                        if club["club"] not in interaction_summary["clubs"]:
                            interaction_summary["clubs"][club["club"].lower()] = {"sentiments": [], "country": "", "sports": [], "reaction sentiments": []}
                        interaction_summary["clubs"][club["club"].lower()]["sentiments"].append(club["sentiment"].lower())
                        interaction_summary["clubs"][club["club"].lower()]["country"] = club["country"].lower()
                        interaction_summary["clubs"][club["club"].lower()]["sports"].append(club["sport"].lower())
                        temp_club_player_sports.add(club["sport"].lower())

                if players:
                    for player in players:   ### v pripade nerozoznania rovnakych klubov-> fuzzy wuzzy + porovnat krajiny
                        if player["player"] not in interaction_summary["players"]:
                            interaction_summary["players"][player["player"].lower()] = {"sentiments": [], "country": "", "sports": "", "reaction sentiments": []}
                        interaction_summary["players"][player["player"].lower()]["sentiments"].append(player["sentiment"].lower())
                        interaction_summary["players"][player["player"].lower()]["country"] = player["country"].lower()
                        interaction_summary["players"][player["player"].lower()]["sport"] = player["sport"].lower()
                        temp_club_player_sports.add(player["sport"].lower())

                if reaction.content["reaction"]["sport"]:
                    for sport in reaction.content["reaction"]["sport"]["sports"]:
                        if sport["sport"].lower() not in interaction_summary["sports"]:
                            interaction_summary["sports"][sport["sport"].lower()] = {"sentiments": [], "c/p tweets": 0, "reaction sentiments": []}  ### sentiments su iba pre tie sporty, ktore su spomenute individualne
                        if sport["sport"].lower() in temp_club_player_sports:
                            interaction_summary["sports"][sport["sport"].lower()]["c/p tweets"] += 1
                        else:
                            interaction_summary["sports"][sport["sport"].lower()]["sentiments"].append(sport["sentiment"].lower())

                og_tweet = source_tweet_content
                if og_tweet["sport"]:
                    if og_tweet["sport"]["sports"]:
                        for sport in og_tweet["sport"]["sports"]:
                            if sport["sport"].lower() not in interaction_summary["sports"]:
                                interaction_summary["sports"][sport["sport"].lower()] = {"sentiments": [], "c/p tweets": 0, "reaction sentiments": []}  ### sentiments su iba pre tie sporty, ktore su spomenute individualne
                            interaction_summary["sports"][sport["sport"].lower()]["reaction sentiments"].append(reaction_translate[reaction.content["reaction_sentiment"]][str(sport["sentiment"]).lower()])

                    if og_tweet["sport"]["clubs"]:
                        for club in og_tweet["sport"]["clubs"]:
                            if club["club"].lower() not in interaction_summary["clubs"]:
                                interaction_summary["clubs"][club["club"].lower()] = {"sentiments": [], "country": "", "sports": [], "reaction sentiments": []}
                                interaction_summary["clubs"][club["club"].lower()]["country"] = club["country"].lower()
                                interaction_summary["clubs"][club["club"].lower()]["sports"].append(club["sport"].lower())
                            interaction_summary["clubs"][club["club"].lower()]["reaction sentiments"].append(reaction_translate[reaction.content["reaction_sentiment"]][club["sentiment"].lower()])

                    if og_tweet["sport"]["players"]:
                        for player in og_tweet["sport"]["players"]:
                            if player["player"].lower() not in interaction_summary["players"]:
                                interaction_summary["players"][player["player"].lower()] = {"sentiments": [], "country": "", "sports": [], "reaction sentiments": []}
                                interaction_summary["players"][player["player"].lower()]["country"] = player["country"].lower()
                                interaction_summary["players"][player["player"].lower()]["sport"] = player["sport"].lower()
                            interaction_summary["players"][player["player"].lower()]["reaction sentiments"].append(reaction_translate[reaction.content["reaction_sentiment"]][player["sentiment"].lower()])


            if type == "music" or source_tweet_content["type"] == "music":
                temp = set()
                if artists:
                    for artist in artists:   ### v pripade nerozoznania rovnakych klubov-> fuzzy wuzzy + porovnat krajiny
                        if artist["artist"] not in interaction_summary["artists"]:
                            interaction_summary["artists"][artist["artist".lower()]] = {"sentiments": [], "country": "", "genres": "", "reaction sentiments": []}
                        interaction_summary["artists"][artist["artist"].lower()]["sentiments"].append(artist["sentiment"].lower())
                        interaction_summary["artists"][artist["artist"].lower()]["country"] = artist["country"].lower()
                        interaction_summary["artists"][artist["artist"].lower()]["genres"].append(artist["genre"].lower())
                        temp.add(player["genre"].lower())
                
                if reaction.content["reaction"]["music"]:
                    for genre in reaction.content["reaction"]["music"]["genres"]:
                        if genre["genre"].lower() not in interaction_summary["genres"]:
                            interaction_summary["genres"][genre["genre"].lower()] = {"sentiments": [], "artist tweets": 0, "reaction sentiments": []}  ### sentiments su iba pre tie sporty, ktore su spomenute individualne
                        if genre["genre"].lower() in temp:
                            interaction_summary["genres"][genre["genre"].lower()]["artist tweets"] += 1
                        else:
                            interaction_summary["genres"][genre["genre"].lower()]["sentiments"].append(genre["sentiment"].lower())

                og_tweet = source_tweet_content
                if og_tweet["music"]:
                    if og_tweet["music"]["genres"]:
                        for genre in og_tweet["music"]["genres"]:
                            if genre["genre"].lower() not in interaction_summary["genres"]:
                                interaction_summary["genres"][genre["genre"].lower()] = {"sentiments": [], "artist tweets": 0, "reaction sentiments": []}  ### sentiments su iba pre tie sporty, ktore su spomenute individualne
                            interaction_summary["genres"][genre["genre"].lower()]["reaction sentiments"].append(reaction_translate[reaction.content["reaction_sentiment"]][genre["sentiment"].lower()])
                    if og_tweet["music"]["artists"]:
                        for artist in og_tweet["music"]["artists"]:
                            if artist["artist"].lower() not in interaction_summary["artists"]:
                                interaction_summary["artists"][artist["artist"].lower()] = {"sentiments": [], "country": "", "genres": [], "reaction sentiments": []}
                                interaction_summary["artists"][artist["artist"].lower()]["country"] = artist["country"].lower()
                                interaction_summary["artists"][artist["artist"].lower()]["genre"] = artist["genre"].lower()
                            interaction_summary["artists"][artist["artist"].lower()]["reaction sentiments"].append(reaction_translate[reaction.content["reaction_sentiment"]][artist["sentiment"].lower()])

        return interaction_summary

    def interest_sum(self, cache):
        data = {}
        with open(cache, 'r', encoding="utf-8") as file:
            cached_profiles = json.load(file)

        for username, analysis in cached_profiles.items():
            if username in self.following:
                data[username] = analysis

        return PROFILE_AI_ANALYSER.profiles_summary(data)
        

'''
Nodes
    Tweety / Reposty
        Politika
            {Ideologia: pocet_krat}   zoradeny list
            
        Sport
            {sports: sentiment_index}    zl
            {clubs: sentiment_index}     zl
            {players: sentiment_index}   zl
            {countries of origin: {clubs: number, players: number}}
            
        Hudba
            {genres, country: sentiment_index}
            {artists: sentiment_index}

        Pomer jazykov
        
    Quote/Comment
        Politika
            {Ideologia: pocet_krat}   zoradeny list
            +Reackie ideologia       ak disagree na napr na lib-left og tweet, tak napise sem

        Sport
            {sports: sentiment_index}    zl
            {clubs: sentiment_index}     zl
            + reakcie
            {players: sentiment_index}   zl
            + reakcie
            {countries of origin: {clubs: number, players: number}}
            
        Hudba
            {genres, country: sentiment_index}
            {artists: sentiment_index}
            + reakcie

        Pomer jazykov
    
    Profily
        to iste iba suhrn

    Hashtagy
            
        
    MetaData
        Bio
        Location
        Profession
        Datum zalozenia
        
    
    => vydedukovat politicku orientaciu
        - sledovane          priemer politickych orientacii sledovanych profilov
        - vyjadrene          priemer pol. or. tweetov, repostov, pripadne hashtagov
        - interagovane na    priemer p.o. quote a comment podla agree/disagree
        => suhrn
    => na kazdy spominany sport nazor
        - -,,-
    => 3 kategorie klubov podla oblubenosti
        
    => 3 kategorie hracov podla oblubenosti
    => na kazdy spominany zaner nazor
    => 3 kategorie hudobnikov podla oblubenosti


Edges

'''



### repost retweet hashtagy 
class Tweet:
    def __init__(self, status_id, username, text, type, source_tweet=None, source_username=None, hashtags=[], mentions=[]):
        self.status_id = status_id
        self.username = username
        self.text = text
        self.type = type
        self.source_tweet = source_tweet
        self.source_username = source_username
        self.hashtags = hashtags
        self.mentions = mentions
        self.date = None
        self.content = None

    def analyse(self):     ###   analyza vynechava niektore,  mozno je to ze v threadoch
        ##########    riesenie => ak najde tweet bez contentu, retrospektivne ho analyzuje
        with open("tweet_analysis.json", 'r', encoding="utf-8") as file:
            cached_tweets = json.load(file)

##        if "1877127950253822340" == self.status_id:
##            print("\n\n\n ANO \n\n\n", self.status_id in cached_tweets, self.status_id in ALL_TWEETS, "\n\n\n")
        
        if self.status_id in cached_tweets:
            self.content = cached_tweets[self.status_id][-1]
            return

        elif self.type == "tweet":
            self.content = TWEET_AI_ANALYSER.analyze_tweet(self.text)
            print(self.type, '\n', self.source_tweet, '\n', self.text, '\n', self.content, '\n')
            cached_tweets[self.status_id] = [self.username, self.text, self.content]
               
        elif self.type == "repost":
            if ALL_TWEETS.get(self.source_tweet, None):
                self.content = TWEET_AI_ANALYSER.analyze_tweet(ALL_TWEETS[self.source_tweet].text)
                print(self.type, '\n', self.source_tweet, '\n', self.text, '\n', self.content, '\n')
                cached_tweets[self.status_id] = [self.username, self.text, self.content]
                 
        elif self.type == "comment":  ### problem je ziskavanie komentarov a threafdy
##            print(self.status_id, self.text, self.source_tweet)# pozriet ci komenty maju spravne source id
            if ALL_TWEETS.get(self.source_tweet, None):
                self.content = TWEET_AI_ANALYSER.analyze_reaction(self.text, ALL_TWEETS[self.source_tweet].text)
                print(self.type, '\n', self.source_tweet, '\n', self.text, '\n', self.content, '\n')
                cached_tweets[self.status_id] = [self.username, self.source_tweet, self.text, self.content]
            else:
                self.content = TWEET_AI_ANALYSER.analyze_tweet(ALL_TWEETS[self.status_id].text)
                print(self.type, '\n', self.source_tweet, '\n', self.text, '\n', self.content, '\n')
                cached_tweets[self.status_id] = [self.username, self.text, self.content]
                
        elif self.type == "quote":
            if ALL_TWEETS.get(self.source_tweet, None):
                self.content = TWEET_AI_ANALYSER.analyze_reaction(self.text, ALL_TWEETS[self.source_tweet].text)
                print(self.type, '\n', self.source_tweet, '\n', self.text, '\n', self.content, '\n')
                cached_tweets[self.status_id] = [self.username, self.source_tweet, self.text, self.content]
            else:
                self.content = TWEET_AI_ANALYSER.analyze_tweet(ALL_TWEETS[self.status_id].text)
                print(self.type, '\n', self.source_tweet, '\n', self.text, '\n', self.content, '\n')
                cached_tweets[self.status_id] = [self.username, self.text, self.content]
                


        with open("tweet_analysis.json", "w", encoding="utf-8") as f:
            json.dump(cached_tweets, f, indent=4, ensure_ascii=False)
                


    def __repr__(self):
        return f"""
            SOURCE ID: {self.status_id}
            USER: {self.username}
            TEXT: {self.text}
            TYPE: {self.type}
            SOURCE TWEET: {self.source_tweet}
            SOURCE USER: {self.source_username}
            HASHTAGS: {self.hashtags}
            MENTIONS: {self.mentions}
            DATE: {self.date}
            CONTENT: {self.content}

        """
    def __str__(self):
        return self.text

    def get_type(self):
        return self.type

    




        







#TYPES = ["entity_centered", "profile_centered"]  # mozno hashtag centered,

#ARGS = depth, time_interval, minimal_edge_weight

class SocialBubble:
    def __init__(self, username, type, **args):
        self.time_interval = args.get("time_interval", None)
        self.type = type
        self.minimal_edge_weight = args.get("minimal_edge_weight", 0)
        if self.type == "entity_centered":
            self.depth = 1
        else:
            self.depth = args["depth"]
        self.username = username
        self.edges = []
        self.nodes = {}  ### name : Node
        self.followed_outside_bubble = {}  ### sledovane profily ucastnikmi bubliny ktore sa v nej nenachadzaju
        self.following_outside_bubble = {} ### sledujuci profilov bubliny ktori nie su v nej
        

    def exist_edge(self, node1: Node, node2: Node):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 or edge.node1 == node2 and edge.node2 == node1:
                return edge
        return None    

    def create_graph(self):
        if self.type == "profile_centered":
            base = Node(self.username)
            self.nodes[self.username] = base
            visited = []
            queue = [base]

            ### vytvori graph friends do urcitej hlbky
            for iteration in range(self.depth):
                new_queue = []
                for node in queue:
                    for friend in node.profile.friends:
                        if friend in visited:
                            continue
                        n = self.nodes.get(friend, None)
                        if not n:
                            n = Node(friend)
                        e = Edge("friends", node, n)
                        new_queue.append(n)
                        self.nodes[friend] = n
                        self.edges.append(e)
                        n.edges.append(e)
                        node.edges.append(e)
                        
                    visited.append(node.profile.username)
                    
                queue = new_queue

            ### doplni chybajuce friendship edges
            for node in queue:
                for friend in node.profile.friends:
                    if friend in self.nodes.keys():
                        edge = self.exist_edge(node, self.nodes[friend])
                        if not edge:
                            e = Edge("friends", node, self.nodes[friend])
                            self.edges.append(e)
                            node.edges.append(e)
                            self.nodes[friend].edges.append(e)

                         
            ### zamedzuje vynechaniu ludi ktory napr nefollowli naspat
            for name, node in self.nodes.items():
                for following in list(set(node.profile.following) - set(node.profile.friends)):
                    if following in self.nodes.keys():
                        ### pridat jednosmerny edge do grafu
                        if not self.exist_edge(node, self.nodes[following]):
                            self.edges.append(Edge("->", node, self.nodes[following]))
                    else:
                        ### pridat following do struktury kde sa spocita kym je sledovana
                        if following not in self.followed_outside_bubble:
                            self.followed_outside_bubble[following] = []
                        self.followed_outside_bubble[following].append(name)
                        
                for follower in list(set(node.profile.followers) - set(node.profile.friends)):
                    if follower in self.nodes.keys():
                        if not self.exist_edge(node, self.nodes[follower]):
                            self.edges.append(Edge("<-", node, self.nodes[follower]))
                    else:
                        ### pridat following do struktury kde sa spocita kym je sledovana
                        if follower not in self.following_outside_bubble:
                            self.following_outside_bubble[follower] = []
                        self.following_outside_bubble[follower].append(name)

            for name in list(set(self.following_outside_bubble.keys()) & set(self.followed_outside_bubble.keys())):
                self.nodes[name] = Node(name)
                for name2 in self.following_outside_bubble[name]:
                    if not self.exist_edge(self.nodes[name], self.nodes[name2]):
                        e = Edge("<-", self.nodes[name], self.nodes[name2])
                        self.edges.append(e)
                        self.nodes[name].edges.append(e)
                        self.nodes[name2].edges.append(e)
                    
                for name2 in self.followed_outside_bubble[name]:
                    if not self.exist_edge(self.nodes[name], self.nodes[name2]):
                        e = Edge("->", self.nodes[name], self.nodes[name2])
                        self.edges.append(e)
                        self.nodes[name].edges.append(e)
                        self.nodes[name2].edges.append(e)
                
            ### spracovat mentions
            self.mentioned_outside_bubble = {}
            for username, node in self.nodes.items():
                for mention, quantity in node.profile.all_mentions.items():
                    if mention in self.nodes.keys():
                        e = self.exist_edge(self.nodes[mention], node)
                        if not e:
                            e = Edge("X", self.nodes[mention], node)
                            self.edges.append(e)
                            node.edges.append(e)
                            self.nodes[mention].edges.append(e)
                        direction = e.direction(self.nodes[mention], node)
                        e.weight["mentions"][direction] += quantity
                            
                    else:
                        self.mentioned_outside_bubble[mention] = username

                   
            ### spracovat reakcie tweetove
            queue = list(self.nodes.values())
            self.interacted_outside_bubble = []
            while queue:
                node = queue.pop(0)
                for tweet in node.profile.reposts + node.profile.comments + node.profile.quotes:
                    node2 = self.nodes.get(tweet.source_username, None)
                    edge = self.exist_edge(node, node2)
                    if edge:
                        direction = edge.direction(node, node2)
                        edge.weight["reactions"][direction].append(tweet)
                    else:
                        if node2:
                            e = self.exist_edge(node, node2)
                            if not e:
                                e = Edge('X', node, node2)
                                self.edges.append(e)
                                node.edges.append(e)
                                node2.edges.append(e)
                            e.weight["reactions"][e.direction(node, node2)].append(tweet)
                        else:
                            n = Node(tweet.source_username)
                            mentions = set(n.profile.all_mentions.keys()) & set(self.nodes.keys())
                            reactions = {}
                            for i in n.profile.reposts + n.profile.quotes + n.profile.comments:
                                if i.source_username in self.nodes.keys():
                                    reactions[i.source_username] = i
                            follows = set(n.profile.following) & set(self.nodes.keys())
                            if mentions or reactions or follows:
                                self.following_outside_bubble.pop(tweet.source_username, None)
                                self.followed_outside_bubble.pop(tweet.source_username, None)
                                self.mentioned_outside_bubble.pop(tweet.source_username, None)
                                self.nodes[tweet.source_username] = n
                                queue.append(n)
                                e = self.exist_edge(n, node)
                                if not e:
                                    e = Edge('X', n, node)
                                    n.edges.append(e)
                                    node.edges.append(e)
                                    self.edges.append(e)
                                e.weight["reactions"][e.direction(node, n)].append(tweet)
                                
                                for i in list(follows):
                                    if not self.exist_edge(n, self.nodes[i]):
                                        e = Edge("->", n, self.nodes[i])
                                        n.edges.append(e)
                                        self.nodes[i].edges.append(e)
                                        self.edges.append(e)
                                    
                                for i,j in reactions.items():
                                    temp = self.exist_edge(n, self.nodes[i])
                                    if temp:
                                        e = temp
                                    else:
                                        e = Edge("X", n, self.nodes[i])
                                        n.edges.append(e)
                                        self.nodes[i].edges.append(e)
                                        self.edges.append(e)
                                    e.weight["reactions"][e.direction(n, self.nodes[i])].append(j)
                                
                                for i in mentions:
                                    temp = self.exist_edge(n, self.nodes[i])
                                    if temp:
                                        e = temp
                                    else:
                                        e = Edge("X", n, self.nodes[i])
                                        n.edges.append(e)
                                        self.nodes[i].edges.append(e)
                                        self.edges.append(e)
                                    e.weight["mentions"][e.direction(n, self.nodes[i])] += n.profile.all_mentions[i]

                                for i in self.mentioned_outside_bubble.get(tweet.source_username, []):
                                    temp = self.exist_edge(n, self.nodes[i])
                                    if temp:
                                        e = temp
                                    else:
                                        e = Edge("X", n, self.nodes[i])
                                        n.edges.append(e)
                                        self.nodes[i].edges.append(e)
                                        self.edges.append(e)
                                    e.weight["mentions"][e.direction(self.nodes[i], n)] += self.nodes[i].all_mentions[tweet.source_username]




                                ''' prekontrolovat ci followers/ing dole'''




                                for i in list(set(n.profile.followers) & set(self.nodes.keys())):
                                    temp = self.exist_edge(n, self.nodes[i])
                                    if temp:
                                        e = temp
                                    else:
                                        e = Edge("X", n, self.nodes[i])
                                        n.edges.append(e)
                                        self.nodes[i].edges.append(e)
                                        self.edges.append(e)
                                    if e.direction(self.nodes[i], n) == "1->2":
                                        direction = '->'
                                    elif e.direction(self.nodes[i], n) == "2->1":
                                        direction = '<-'
                                    else:
                                        raise ValueError("nejaka chyba")
                                    
                                    if e.weight["follows"] == 'X' or e.weight["follows"] == direction:
                                        e.weight["follows"] = direction
                                    else:
                                        e.weight["follows"] = "friends"
                                                    

                            else:
                                self.interacted_outside_bubble.append((tweet.source_username, tweet))
                        
            #print(self.followed_outside_bubble, self.mentioned_outside_bubble)
                
            ## pridat interagovane profily mimo bubliny do nejakeho zoznamu, zoradene podla poctu interakcii clenov bubliny a followov


            ### ak je repost/comment/quote a og tweet autor sleduje niekoho v bubline tak je pridany





                            
            
                
            

                
                
        elif self.type == "entity_centered":
            base = Node(self.username)
            self.nodes[self.username] = base
            for follower in base.profile.followers:
                n = Node(follower)
                e = Edge("friends", base, n)
                self.nodes[follower] = n
                self.edges.append(e)
                n.edges.append(e)
                base.edges.append(e)
                for i in list(set(n.profile.friends) & set(self.nodes.keys()) - set(self.username)):
                    e2 = Edge("friends", self.nodes[i], n)
                    self.edges.append(e2)
                    self.nodes[i].edges.append(e2)
                    n.edges.append(e2)
            ### najdenie vzajomnych prepojeni




        ### prejde follows ludi v bubline, hlada bublina & (follows - friends), spocita co je kym followovane pri profiloch mimo bubliny
        ### prejde tweety, pozrie ci su nejake retweety


    def visualize_graph(self):
        G = nx.DiGraph()  # Use a directed graph for arrows

        # Add nodes
        for node in self.nodes.values():
            G.add_node(node.profile.username)
            for i in node.profile.tweets + node.profile.reposts + node.profile.comments + node.profile.quotes:
                repr(i)
##        for edge in self.edges:
##            print(edge)

        # Define edge lists for different colors
        red_edges = []  # For "->" or "<-"
        green_edges = []  # For "friends"
        blue_edges = []  # For "X"

        # Add edges with color classification
        for edge in self.edges:
            node1 = list(edge.directions.keys())[0]
            node2 = edge.get_second_node(node1)
            follow_type = edge.weight.get("follows", "friends")

            if follow_type == "->" or follow_type == "<-":
                red_edges.append((node1.profile.username, node2.profile.username))
            elif follow_type == "friends":
                green_edges.append((node1.profile.username, node2.profile.username))
            elif follow_type == "X":
                blue_edges.append((node1.profile.username, node2.profile.username))

        # Draw the graph
        plt.figure(figsize=(10, 8))

        # More spread-out nodes
        pos = nx.spring_layout(G, seed=42, k=1.2, iterations=100, scale=1)

        # Draw nodes
        nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, font_size=10)

        # Draw edges with different colors
        nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color="red", width=2, arrows=True, arrowsize=15)
        nx.draw_networkx_edges(G, pos, edgelist=green_edges, edge_color="green", width=2)
        nx.draw_networkx_edges(G, pos, edgelist=blue_edges, edge_color="blue", width=2)

        plt.title("Social Graph Visualization (With Colored Edges)")
        plt.show()

    def get_outside_profiles_data(self, follower_count):
        outside_bubble_data = {}
        for node in self.nodes.values():
            for profile in node.profile.following_outside_bubble_big_profiles(self.followed_outside_bubble, follower_count):
                a,b,c = profile
                if a not in outside_bubble_data.keys():
                    outside_bubble_data[a] = (b,c)
        return outside_bubble_data
            
    def tweet_analysis(self):
        for tweet in ALL_TWEETS.values():
            tweet.analyse()


    def profile_analysis(self, profiles, cache="profile_analysis.json"):
        global OUTSIDE_BUBBLE_PROFILES_ANALYSED
        with open(cache, 'r', encoding="utf-8") as file:
            cached_profiles = json.load(file)

        #print(profiles)

        serpapi = AIAnalysis.SerpAPI()

        for username, profile in profiles.items(): #(screen_name: name, description)
            if username in cached_profiles.keys():
                continue
            entity_data = serpapi.get_entity(profile[0])
            serpapi_formated = serpapi.process_entity(entity_data)
            serpapi_formated["Twitter username"] = username
            serpapi_formated["Twitter bio"] = profile[1]
            serpapi_formated["name"] = profile[0]
            analysis = PROFILE_AI_ANALYSER.analyze_profile_II(serpapi_formated)
            analysis["full_name"] = profile[0]
            cached_profiles[username] = analysis
            print(username, analysis)


        OUTSIDE_BUBBLE_PROFILES_ANALYSED = cached_profiles
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(cached_profiles, f, indent=4, ensure_ascii=False)


    def profiles_summary(self, time_interval=None, steps=1):
        sums = {}
        for username, node in self.nodes.items():
            node.profile.summary()



            

        




            
        

''' hrany s interakciami ohladom konkretnej temy '''



SB = SocialBubble("tofaaakt", "profile_centered", depth=0)

SB.create_graph()

SB.visualize_graph()

opd = SB.get_outside_profiles_data(THRESHOLD)

##print(len(opd))

##outside_bubble_data={}
##for node in SB.nodes.values():
##    for profile in node.profile.following_outside_bubble_big_profiles(SB.followed_outside_bubble, 2000):
##        a,b,c = profile
##        if a not in outside_bubble_data.keys():
##            outside_bubble_data[a] = (b,c)
##
##a = set(outside_bubble_data)- set(SB.nodes["jarro01"].profile.following) - set(SB.nodes["pushkicknadusu"].profile.following)
##
##filtered_dict = {k: v for k, v in outside_bubble_data.items() if k in a}
##
##print(filtered_dict)

SB.profile_analysis(opd)

SB.tweet_analysis()

SB.profiles_summary()


'''
1. VYTVORIT KVALITNU ANALYZU PROFILU + VYOBRAZENIE


mozne funkcie:
    UNION, INTERSECTION
    AK JE NEJAKA TOPIC, ZVYRAZNIT TIE KTORYCH SA TO TYKA + HRANY (INTERAKCIA, ROVNAKY FOLLOW NA DANU TOPIC ...)


'''







#############   AK SERPAPI NEVRATI KVALITNY OUTPUT, PREFEROVAT BIO

#############   SKONTROLOVAT CI DESCRIPTION == BIO

#############   BUDE TREBA SKONTROLOVAT IMPLEMENTACIU CREATE_BUBBLE, CI
#############   NIEKDE NIE SU VYNECHANE UDAJE ATD
