import copy
import networkx as nx
import matplotlib.pyplot as plt
import test_twitter_scrapper_from_json
import AIAnalysis
import json
from fuzzywuzzy import process, fuzz
import math
from collections import Counter
import numpy as np
from flask import Flask, render_template_string
from datetime import datetime, time
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go


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
MONTH_MAP = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
IDEOLOGIES_ANGLES = {"liberalism": 225, "nationalism": 45,"conservatism": 15,"socialism": 135,"communism": 135,"environmentalism": 145,"social democracy": 160,"progressivism": 195,"anarchism": 270,"centrism": None,"libertarianism": 315,"fascism": 70,"authoritarianism": 90,"religious-based ideology": 45}


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
    def __init__(self, username, expression, interaction, interest, hashtags, mentions, location, avg_activity, daily_activity, interval):
        self.username = username
        self.expression_sum = expression
        self.interaction_sum = interaction
        self.interest_sum = interest
        self.overall = self.overall_sum()
        self.top_hashtags = sorted(hashtags.keys(), key=lambda k: hashtags[k], reverse=True)[:5]
        self.top_mentions = sorted(mentions.keys(), key=lambda k: mentions[k], reverse=True)[:5]
        self.location = location
        self.avg_activity = avg_activity
        self.daily_activity = {}
        self.interval = interval
        self.filename = "graphs.html"
        for hour, tweets in daily_activity.items():
            self.daily_activity[hour] = len(tweets)
        self.languages = {}
        for lg in expression["languages"] + interaction["languages"]:
            if lg not in self.languages:
                self.languages[lg] = 0
            self.languages[lg] += 1

        
##    def interpret_sentiment_list(self, sentiments, pos, neu, neg):
##        mapping = {'positive': pos, 'neutral': neu, 'negative': neg}
##        scores = [mapping[s] for s in sentiments]
##        raw_score = sum(scores) / len(scores)
##        confidence = min(1.0, math.log2(len(scores) + 1) / 3)
##        return raw_score * confidence
        

    def interpret_sentiment_list(self, sentiments, pos, neu, neg):
        mapping = {'positive': pos, 'neutral': neu, 'negative': neg}
        scores = [mapping[s] for s in sentiments]
        
        if not scores:
            return 0  # Avoid division by zero

        raw_score = sum(scores) / len(scores)

        n = len(scores)
        confidence = n / (n + 3.5)  # Slower start, then grows faster

        return raw_score * confidence

    def show(self):
        self.generate_html(False)

    def get_summary(self):
        return {
                "username": self.username,
                "expression_sum": self.expression_sum,
                "interaction_sum": self.interaction_sum,
                "interest_sum": self.interest_sum,
                "overall": self.overall,
                "top_hashtags": self.top_hashtags,
                "top_mentions": self.top_mentions,
                "location": self.location,
                "avg_activity": self.avg_activity,
                "daily_activity": self.daily_activity,
                "interval": self.interval,
                "languages": self.languages,
                "compass": self.compass,
                "topic_counts": self.topic_counts
            }
        
    def __str__(self):
        return f'''
                {self.username},

                HASHTAGS: {self.top_hashtags}
                MENTIONS: {self.top_mentions}
                LOCATION: {self.location}
                AVG ACTIVITY: {self.avg_activity}
                DAILY ACTIVITY: {self.daily_activity}

                SUM: {self.overall}
                LANGUAGES: {self.languages}
            '''

    def overall_sum(self):
        ### SPORT
        ''' poradie podla weight h->l: expression >= interaction >(o trochu) interest '''
        self.sports = {}
        self.clubs = {}
        self.athletes = {}
        self.genres = {}
        self.artists = {}
        self.others = {}
        self.politics = {k: {"sentiment": 0, "mentions": {"ex/int": 0, "interest": 0}} for k in ["liberalism", "nationalism", "conservatism", "socialism", "communism", "environmentalism", "social democracy", "progressivism", "anarchism", "centrism", "libertarianism", "fascism", "authoritarianism", "religious-based ideology"]}
        self.sport_countries = {}
        self.music_countries = {}

        for sport, data in self.expression_sum["sports"].items():
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.5)
            mentions = len(data["sentiments"]) + data["c/p tweets"]
            self.sports[sport] = {"sentiment": [sentiment], "mentions": {"expression": mentions, "interaction": 0, "interest": 0}}

        for sport, data in self.interaction_sum["sports"].items():
            sentiment = self.interpret_sentiment_list(self.data["sentiments"], 1.1, 0.4, -0.5)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + data["c/p tweets"] + len(data["reaction sentiments"])
            temp = process.extractOne(sport, self.sports.keys()) or [0,0]
            if sport not in self.sports and temp[1] < 85:
                self.sports[sport.lower()] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": mentions, "interest": 0}}
            else:
                self.sports[temp[0]]["sentiment"].append(sentiment)
                self.sports[temp[0]]["sentiment"].append(reaction_sentiment)
                self.sports[temp[0]]["mentions"]["interaction"] += mentions

        if self.interest_sum["sport"]:
            for sport in self.interest_sum["sport"]:
                sentiment = self.interpret_sentiment_list(["positive"]*sport["counter"], 0.6, 0, 0)
                temp = process.extractOne(sport["sport"].lower(), self.sports.keys()) or [0,0]
                if sport["sport"].lower() not in self.sports and temp[1] < 85:
                    self.sports[sport["sport"].lower()] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": 0, "interest": sport["counter"]}}
                else:
                    self.sports[temp[0]]["sentiment"].append(sentiment)
                    self.sports[temp[0]]["mentions"]["interest"] += sport["counter"]
                

        for club, data in self.expression_sum["clubs"].items():
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.7)
            mentions = len(data["sentiments"])
            self.clubs[club] = {"sentiment": [sentiment], "mentions": {"expression": mentions, "interaction": 0, "interest": 0}}
            for sport in list(set(data["sports"])):
                if sport.lower() not in self.sport_countries:
                    self.sport_countries[sport.lower()] = {}
                if data["country"].lower() not in self.sport_countries[sport.lower()]:
                    self.sport_countries[sport.lower()][data["country"].lower()] = 0
                self.sport_countries[sport.lower()][data["country"].lower()] += 1
                

        for club, data in self.interaction_sum["clubs"].items():
            sentiment = self.interpret_sentiment_list(self.data["sentiments"], 1.1, 0.4, -0.7)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + len(data["reaction sentiments"])
            temp = process.extractOne(club, self.clubs.keys()) or [0,0]

            sport = self.expression_sum.get("clubs", {}).get(temp[0], None)["sports"] or self.interaction_sum.get("clubs", {}).get(temp[0], None)["sports"] or self.interest_sum.get("clubs", {}).get(temp[0], None)["sports"]
            if sport is None:
                print("NO SPORT")
                self.clubs[club] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": mentions, "interest": 0}}
            else:
                sport = Counter(sport).most_common(1)[0][0]
                
            if club not in self.clubs and temp[1] < 85:
                self.clubs[club] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": mentions, "interest": 0}}
                
            elif self.clubs[temp[0]] and fuzz.ratio(sport, Counter(data["sports"]).most_common(1)[0][0]) > 87:
                self.clubs[temp[0]]["sentiment"].append(sentiment)
                self.clubs[temp[0]]["sentiment"].append(reaction_sentiment)
                self.clubs[temp[0]]["mentions"]["interaction"] += mentions
                
            else:
                self.clubs[club] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": mentions, "interest": 0}}

            for sport in list(set(data["sports"])):
                if sport.lower() not in self.sport_countries:
                    self.sport_countries[sport.lower()] = {}
                if data["country"].lower() not in self.sport_countries[sport.lower()]:
                    self.sport_countries[sport.lower()][data["country"].lower()] = 0
                self.sport_countries[sport.lower()][data["country"].lower()] += 1
            
                

        if self.interest_sum["sport"]:
            for sport in self.interest_sum["sport"]:
                if sport["countries"]:
                    for country in sport["countries"]:
                        if sport["sport"].lower() not in self.sport_countries:
                            self.sport_countries[sport["sport"].lower()] = {}
                        if country["country"].lower() not in self.sport_countries[sport["sport"].lower()]:
                            self.sport_countries[sport["sport"].lower()][country["country"].lower()] = 0
                        self.sport_countries[sport["sport"].lower()][country["country"].lower()] += len(country["clubs"] or []) + len(country["athletes"] or [])
                        
                        if country["clubs"]:
                            for club in set(country["clubs"]):
                                if self.clubs:
                                    temp = process.extractOne(club.lower(), self.clubs.keys())
                                    #print(temp, club)
                                    if temp[1] > 85:
                                        mentions = country["clubs"].count(club)
                                        sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                        self.clubs[temp[0].lower()]["sentiment"].append(sentiment)
                                        self.clubs[temp[0].lower()]["mentions"]["interest"] += mentions
                                    else:
                                        mentions = country["clubs"].count(club)
                                        sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                        self.clubs[club.lower()] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": 0, "interest": mentions}}
                                        
                                elif club.lower() not in self.clubs:
                                    mentions = country["clubs"].count(club)
                                    sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                    self.clubs[club.lower()] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": 0, "interest": mentions}}

            
        for player, data in self.expression_sum["players"].items():
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.7)
            mentions = len(data["sentiments"])
            self.athletes[player] = {"sentiment": [sentiment], "mentions": {"expression": mentions, "interaction": 0, "interest": 0}}
            for sport in list(set(data["sports"])):
                if sport.lower() not in self.sport_countries:
                    self.sport_countries[sport.lower()] = {}
                if data["country"].lower() not in self.sport_countries[sport.lower()]:
                    self.sport_countries[sport.lower()][data["country"].lower()] = 0
                self.sport_countries[sport.lower()][data["country"].lower()] += 1
            

        for player, data in self.interaction_sum["players"].items():
            sentiment = self.interpret_sentiment_list(self.data["sentiments"], 1.1, 0.4, -0.7)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + len(data["reaction sentiments"])
            temp = process.extractOne(sport, self.sports.keys()) or [0,0]
            if player not in self.players and temp[1] < 85:
                self.athletes[player] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": mentions, "interest": 0}}
            else:
                self.athletes[temp[0]]["sentiment"].append(sentiment)
                self.athletes[temp[0]]["sentiment"].append(reaction_sentiment)
                self.athletes[temp[0]]["mentions"]["interaction"] += mentions
                
            for sport in list(set(data["sports"])):
                if sport.lower() not in self.sport_countries:
                    self.sport_countries[sport.lower()] = {}
                if data["country"].lower() not in self.sport_countries[sport.lower()]:
                    self.sport_countries[sport.lower()][data["country"].lower()] = 0
                self.sport_countries[sport.lower()][data["country"].lower()] += 1
            

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
                                        self.athletes[temp[0].lower()]["mentions"]["interest"] += mentions
                                    else:
                                        mentions = country["athletes"].count(athlete)
                                        sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                        self.athletes[athlete.lower()] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": 0, "interest": mentions}}
                                        
                                elif athlete.lower() not in self.athletes:
                                    mentions = country["athletes"].count(athlete)
                                    sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                    self.athletes[athlete] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": 0, "interest": mentions}}

        for genre, data in self.expression_sum["genres"].items():
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.5)
            mentions = len(data["sentiments"]) + data["artist tweets"]
            self.genres[genre] = {"sentiment": [sentiment], "mentions": {"expression": mentions, "interaction": 0, "interest": 0}}

        for genre, data in self.interaction_sum["genres"].items():
            sentiment = self.interpret_sentiment_list(self.data["sentiments"], 1.1, 0.4, -0.5)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + data["artist tweets"] + len(data["reaction sentiments"])
            temp = process.extractOne(genre, self.genres.keys()) or [0,0]
            if genre not in self.genres and temp[1] < 85:
                self.genres[genre.lower()] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": mentions, "interest": 0}}
            else:
                self.genres[temp[0]]["sentiment"].append(sentiment)
                self.genres[temp[0]]["sentiment"].append(reaction_sentiment)
                self.genres[temp[0]]["mentions"]["interaction"] += mentions

        if self.interest_sum["music"]:
            for genre in self.interest_sum["music"]:
                sentiment = self.interpret_sentiment_list(["positive"]*genre["counter"], 0.6, 0, 0)
                temp = process.extractOne(genre["genre"].lower(), self.genres.keys()) or [0,0]
                if genre["genre"].lower() not in self.genres and temp[1] < 85:
                    self.genres[genre["genre"].lower()] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": 0, "interest": genre["counter"]}}
                else:
                    self.genres[temp[0]]["sentiment"].append(sentiment)
                    self.genres[temp[0]]["mentions"]["interest"] += genre["counter"]



        for artist, data in self.expression_sum["artists"].items():
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.7)
            mentions = len(data["sentiments"])
            self.artists[artist] = {"sentiment": [sentiment], "mentions": {"expression": mentions, "interaction": 0, "interest": 0}}
            for genre in list(set(data["genres"])):
                if genre.lower() not in self.music_countries:
                    self.music_countries[genre.lower()] = {}
                if data["country"].lower() not in self.music_countries[genre.lower()]:
                    self.music_countries[genre.lower()][data["country"].lower()] = 0
                self.music_countries[genre.lower()][data["country"].lower()] += 1
            

        for artist, data in self.interaction_sum["players"].items():
            sentiment = self.interpret_sentiment_list(self.data["sentiments"], 1.1, 0.4, -0.7)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + len(data["reaction sentiments"])
            temp = process.extractOne(artist, self.artists.keys()) or [0,0]
            if artist["artist"].lower() not in self.artists and temp[1] < 85:
                self.artists[artist] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": mentions, "interest": 0}}
            else:
                self.artists[artist]["sentiment"].append(sentiment)
                self.artists[artist]["sentiment"].append(reaction_sentiment)
                self.artists[artist]["mentions"]["interaction"] += mentions
                
            for genre in list(set(data["genres"])):
                if genre.lower() not in self.music_countries:
                    self.music_countries[genre.lower()] = {}
                if data["country"].lower() not in self.music_countries[genre.lower()]:
                    self.music_countries[genre.lower()][data["country"].lower()] = 0
                self.music_countries[genre.lower()][data["country"].lower()] += 1
            

        if self.interest_sum["music"]:
            for genre in self.interest_sum["music"]:
                if genre["genre"].lower() not in self.music_countries:
                    self.music_countries[genre["genre"].lower()] = {}
                for country in genre["countries"] or []:
                    if country.lower() not in self.music_countries[genre["genre"].lower()]:
                        self.music_countries[genre["genre"].lower()][country.lower()] = 0
                    self.music_countries[genre["genre"].lower()][country.lower()] += 1
                
                if genre["artists"]:
                    for artist in set(genre["artists"]):
                        if self.artists:
                            temp = process.extractOne(artist.lower(), self.artists.keys())
                            if temp[1] > 85:
                                mentions = genre["artists"].count(temp[0])
                                sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                self.artists[temp[0].lower()]["sentiment"].append(sentiment)
                                self.artists[temp[0].lower()]["mentions"]["interest"] += mentions
                            else:
                                mentions = genre["artists"].count(artist)
                                sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                                self.artists[artist.lower()] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": 0, "interest": mentions}}
                                
                        elif artist.lower() not in self.artists:
                            mentions = genre["artists"].count(artist)
                            sentiment = self.interpret_sentiment_list(["positive"]*mentions, 0.6, 0, 0)
                            self.artists[artist.lower()] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": 0, "interest": mentions}}

                        
        self.others = copy.deepcopy(self.interest_sum["other_interests"])

        
        for ideology in copy.deepcopy(self.expression_sum["politics"]["type"])+copy.deepcopy(self.interaction_sum["politics"]["type"]):
            if ideology.lower() in ["", "n/a", "none", None, "unknown"]:
                continue
            
            ideology_lower = ideology.lower()
            first8 = ideology_lower[:8]
            best_match = None
            best_score = 0

            for existing_ideology in self.politics.keys():
                existing_first8 = existing_ideology[:8]
                score = fuzz.ratio(first8, existing_first8)
                
                if score > best_score:
                    best_score = score
                    best_match = existing_ideology
                    if best_score == 100:
                        break
            
            if best_match and best_score >= 93:
                self.politics[best_match]["sentiment"] += 1
                self.politics[best_match]["mentions"]["ex/int"] += 1
            else:
                self.politics[ideology_lower] = {
                    "sentiment": 1,
                    "mentions": {
                        "ex/int": 1,
                        "interest": 0
                    }
                }
            
        for ideology in copy.deepcopy(self.expression_sum["politics"]["no type"])+copy.deepcopy(self.interaction_sum["politics"]["no type"]):
            if ideology.lower() in ["centrism", "", "n/a", "none", None, "unknown"]:
                continue
            ideology_lower = ideology.lower()
            first8 = ideology_lower[:8]
            best_match = None
            best_score = 0
            
            for existing_ideology in self.politics.keys():
                existing_first8 = existing_ideology[:8]
                score = fuzz.ratio(first8, existing_first8)
                
                if score > best_score:
                    best_score = score
                    best_match = existing_ideology
                    if best_score == 100:
                        break
            

            if best_match and best_score >= 93:
                self.politics[best_match]["sentiment"] += 1/7
                self.politics[best_match]["mentions"]["ex/int"] += 1/7
            else:
                self.politics[ideology_lower] = {
                    "sentiment": 1/7,
                    "mentions": {
                        "ex/int": 1/7,
                        "interest": 0
                    }
                }

        for ideology in copy.deepcopy(self.interaction_sum["politics"]["type reaction"]):
            if ideology.lower() in ["", "n/a", "none", None, "unknown"]:
                continue
            i = ideology.lstrip("-")
            num = len(ideology) - len(i)
            add = -0.25 * num**2 - 0.5 * num + 1
            ideology_lower = i.lower()
            first8 = ideology_lower[:8]
            best_match = None
            best_score = 0
            
            for existing_ideology in self.politics.keys():
                existing_first8 = existing_ideology[:8]
                score = fuzz.ratio(first8, existing_first8)
                
                if score > best_score:
                    best_score = score
                    best_match = existing_ideology
                    if best_score == 100:
                        break
            

            if best_match and best_score >= 93:
                self.politics[best_match]["sentiment"] += add
                self.politics[best_match]["mentions"]["ex/int"] += add
            else:
                self.politics[ideology_lower] = {
                    "sentiment": add,
                    "mentions": {
                        "ex/int": add,
                        "interest": 0
                    }
                }

        for ideology in copy.deepcopy(self.interaction_sum["politics"]["no type reaction"]):
            if ideology.lower() in ["", "n/a", "none", None, "centrism", "unknown"]:
                continue
            i = ideology.lstrip("-")
            num = len(ideology) - len(i)
            add = -0.25 * num**2 - 0.5 * num + 1
            ideology_lower = i.lower()
            first8 = ideology_lower[:8]
            best_match = None
            best_score = 0
            
            for existing_ideology in self.politics.keys():
                existing_first8 = existing_ideology[:8]
                score = fuzz.ratio(first8, existing_first8)
                
                if score > best_score:
                    best_score = score
                    best_match = existing_ideology
                    if best_score == 100:
                        break
            

            if best_match and best_score >= 93:
                self.politics[best_match]["sentiment"] += add/7
                self.politics[best_match]["mentions"]["ex/int"] += add/7
            else:
                self.politics[ideology_lower] = {
                    "sentiment": add/7,
                    "mentions": {
                        "ex/int": add/7,
                        "interest": 0
                    }
                }

        if self.interest_sum["politics"]:
            for ideology in self.interest_sum["politics"]["ideologies"] or []:
                ideology_lower = ideology["ideology"].lower()
                first8 = ideology_lower[:8]
                best_match = None
                best_score = 0
                
                for existing_ideology in self.politics.keys():
                    existing_first8 = existing_ideology[:8]
                    score = fuzz.ratio(first8, existing_first8)
                    
                    if score > best_score:
                        best_score = score
                        best_match = existing_ideology
                        if best_score == 100:
                            break
                

                if best_match and best_score >= 93:
                    self.politics[best_match]["sentiment"] += ideology["counter"]
                    self.politics[best_match]["mentions"]["ex/int"] += ideology["counter"]
                else:
                    self.politics[ideology_lower] = {
                        "sentiment": ideology["counter"],
                        "mentions": {
                            "ex/int": 0,
                            "interest": ideology["counter"]
                        }
                    }
                
        
        with open("test.json", 'w', encoding="utf-8") as file:
            json.dump([self.sports, self.clubs, self.athletes, self.genres, self.artists], file, indent=4)


        return {"sport":self.sports, "club":self.clubs, "athlete":self.athletes, "genre":self.genres, "artist":self.artists, "politics":self.politics, "other":self.others}


    def create_pie_chart(self, labels, values, title):
        if labels and values:
            pie_chart = go.Figure(go.Pie(labels=labels, values=values, title=title))
            pie_chart.update_layout(width=350, height=350)
            return pie_chart.to_html(full_html=False)

    def create_radar_chart(self, labels, values, title):
        if labels and values:
            radar_chart = go.Figure()
            radar_chart.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself'))
            radar_chart.update_layout(title=title, polar=dict(radialaxis=dict(visible=False)), width=350, height=350)
            return radar_chart.to_html(full_html=False)

    def create_bar_chart(self, labels, values, title):
        if labels and values:
            bar_chart = go.Figure(go.Bar(x=labels, y=values, marker_color=['red' if v < -2 else 'yellow' if -2 <= v <= 2 else 'green' for v in values]))
            bar_chart.update_layout(title=title, yaxis=dict(range=[-10, 10]), width=350, height=350)
            return bar_chart.to_html(full_html=False)

    def create_grid_chart(self, points, grid_size=10, background_colors=['#ff7575','#42aaff','#9aed97','#c09aec'], title="Political Compass"):
        """
        Creates an NxN grid plot with points at specified coordinates.
        
        Args:
            points: List of tuples in format (x, y, label, color)
            grid_size: Size of the grid (default 10 for -5 to +5)
            background_colors: Optional list of colors for quadrants [Q1, Q2, Q3, Q4]
            title: Chart title
        """
        # Create figure
        fig = go.Figure()
        
        # Create grid background with optional quadrant colors
        if background_colors and len(background_colors) >= 4:
            # Add quadrant rectangles
            fig.add_shape(type="rect", x0=-grid_size, y0=0, x1=0, y1=grid_size, 
                         fillcolor=background_colors[0], opacity=0.7, layer="below")
            fig.add_shape(type="rect", x0=0, y0=0, x1=grid_size, y1=grid_size, 
                         fillcolor=background_colors[1], opacity=0.7, layer="below")
            fig.add_shape(type="rect", x0=-grid_size, y0=-grid_size, x1=0, y1=0, 
                         fillcolor=background_colors[2], opacity=0.7, layer="below")
            fig.add_shape(type="rect", x0=0, y0=-grid_size, x1=grid_size, y1=0, 
                         fillcolor=background_colors[3], opacity=0.7, layer="below")
        
        # Add grid lines
        for i in range(-grid_size, grid_size+1):
            fig.add_shape(type="line", x0=i, y0=-grid_size, x1=i, y1=grid_size, 
                         line=dict(color="DarkGray", width=1))
            fig.add_shape(type="line", x0=-grid_size, y0=i, x1=grid_size, y1=i, 
                         line=dict(color="DarkGray", width=1))
        
        # Add axes
        fig.add_shape(type="line", x0=-grid_size, y0=0, x1=grid_size, y1=0, 
                     line=dict(color="Black", width=2))
        fig.add_shape(type="line", x0=0, y0=-grid_size, x1=0, y1=grid_size, 
                     line=dict(color="Black", width=2))
        
        # Add points with hover text
        for x, y, label, color in points:
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                name=label,
                mode='markers+text',
                marker=dict(size=12, color=color),
                text=[label],
                textposition="top center",
                hoverinfo="text",
                hovertext=f"{label}<br>X: {x:.2f}<br>Y: {y:.2f}"
            ))
        
        # Set layout
        fig.update_layout(
            title=title,
            xaxis=dict(range=[-grid_size, grid_size], title="Economic (Left-Right)"),
            yaxis=dict(range=[-grid_size, grid_size], title="Social (Libertarian-Authoritarian)"),
            width=400,
            height=400,
            showlegend=False
        )
        
        return fig.to_html(full_html=False)

    def generate_html(self, do_html=True):
        # === 1. Koláčový graf (jazyky) ===
##        print(self.expression_sum, '\n')
##        print(self.interaction_sum, '\n')
##        print(self.interest_sum, '\n')
##        print(self.overall, '\n')


                # HTML content
        topic_tweet_count = {"politics": len(self.expression_sum["politics"]["type"])+len(self.interaction_sum["politics"]["type reaction"])+len(self.interaction_sum["politics"]["type"]), "sport": 0, "music": 0, "other": 0}
        for sport in list(self.overall["sport"].values()) + list(self.overall["club"].values()) + list(self.overall["athlete"].values()):
            topic_tweet_count["sport"] += sport["mentions"]["expression"] + sport["mentions"]["interaction"]
        for music in list(self.overall["genre"].values()) + list(self.overall["artist"].values()):
            topic_tweet_count["music"] += music["mentions"]["expression"] + music["mentions"]["interaction"]
##########        topic_tweet_count["others"] = "NOT IMPLEMENTED"

        topic_interest_count = {"politics": 0, "sport": 0, "music": 0, "other": 0}
        for politics in list(self.overall["politics"].values()):
            topic_interest_count["politics"] += politics["mentions"]["interest"]
        for sport in list(self.overall["sport"].values()) + list(self.overall["club"].values()) + list(self.overall["athlete"].values()):
            topic_interest_count["sport"] += sport["mentions"]["interest"]
        for music in list(self.overall["genre"].values()) + list(self.overall["artist"].values()):
            topic_interest_count["music"] += music["mentions"]["interest"]
        topic_interest_count["other"] = sum(map(lambda x: x["counter"], self.overall["other"] or []))

        all_keys = topic_tweet_count.keys() | topic_interest_count.keys()  # Union of keys
        topic_all_count = {key: topic_tweet_count.get(key, 0) + topic_interest_count.get(key, 0) for key in all_keys}

        compass_x = sum(
            [
                data["sentiment"] * math.cos(math.radians(IDEOLOGIES_ANGLES.get(name, 90)))
                for name, data in self.overall["politics"].items()
                if name != "centrism"  # Explicitly exclude centrism
            ]
        )

        compass_y = sum(
            [
                data["sentiment"] * math.sin(math.radians(IDEOLOGIES_ANGLES.get(name, 0)))
                for name, data in self.overall["politics"].items()
                if name != "centrism"  # Explicitly exclude centrism
            ]
        )

        

        max_r = sum([data["sentiment"] for name, data in self.overall["politics"].items() if name != "centrism"])
        self.compass = (compass_x, compass_y, max_r)
        self.topic_counts = (topic_tweet_count, topic_interest_count, topic_all_count)
        
        html_content = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Vizualizácia grafov</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; }}
                .container {{ display: flex; flex-direction: column; align-items: center; gap: 20px; }}
                .row {{
                    display: flex;
                    justify-content: flex-start; /* Aligns items to the left */
                    gap: 20px;
                    width: 100%;
                    overflow-x: auto; /* Allows horizontal scrolling if content overflows */
                    white-space: nowrap; /* Prevents items from wrapping to next line */
                    padding: 10px 0; /* Optional: adds some vertical padding */
                }}
                .chart {{
                    flex: 0 0 auto; /* Prevents charts from shrinking/growing */
                    width: 400px; /* Fixed width for each chart */
                    height: 400px; /* Fixed height for each chart */
                }}
            </style>
        </head>
        <body>
            <h1>{self.username} | {self.interval[0]} -> {self.interval[1]}</h1>
            
            <div class="container">
                <h2>Basic info</h2>
                <div class="row">
                    <div class="chart">{self.create_pie_chart(list(self.languages.keys()), list(self.languages.values()), "Jazyky")}</div>
                    <div class="chart">{self.create_pie_chart([f'{i}:00-{i+1}:00' for i in self.daily_activity.keys()], list(self.daily_activity.values()), "Denna aktivita")}</div>
                </div>
                <div class="row">
                    <p>Top hashtagy: {'  '.join([f'#{i}' for i in self.top_hashtags])}</p>
                    <p>Top zmienky: {'  '.join([f'@{i}' for i in self.top_mentions])}</p>
                    <p>Aktivita: {self.avg_activity}</p>
                    <p>Lokacia: {self.location}</p>
                </div>
                <div class="row">
                    <div class="chart">{self.create_pie_chart(list(topic_tweet_count.keys()), list(topic_tweet_count.values()), "Topic tweets")}</div>
                    <div class="chart">{self.create_pie_chart(list(topic_interest_count.keys()), list(topic_interest_count.values()), "Topic interests")}</div>
                    <div class="chart">{self.create_pie_chart(list(topic_all_count.keys()), list(topic_all_count.values()), "Topic overall")}</div>
                </div>
                <h2>Sport info</h2>
                <h4>Sports</h4>
                <div class="row">
                    <div class="chart">{self.create_radar_chart(list(self.overall["sport"].keys()), [sum(x["mentions"].values()) for x in self.overall["sport"].values()], "Interest")}</div>
                    <div class="chart">{self.create_bar_chart(list(self.overall["sport"].keys()), [round(sum(x["sentiment"])*10, 2) for x in self.overall["sport"].values()], "Sentiment")}</div>
                </div>
                <div class="row">
                    {" ".join([f'<div class="chart">{self.create_pie_chart(list(data.keys()), list(data.values()), sport)}</div>' for sport,data in self.sport_countries.items()])}
                </div>
                <h4>Clubs</h4>
                <div class="row">
                    <div class="chart">{self.create_radar_chart(list(self.overall["club"].keys()), [sum(x["mentions"].values()) for x in self.overall["club"].values()], "Interest")}</div>
                    <div class="chart">{self.create_bar_chart(list(self.overall["club"].keys()), [round(sum(x["sentiment"])*10, 2) for x in self.overall["club"].values()], "Sentiment")}</div>
                </div>
                <h4>Athletes</h4>
                <div class="row">
                    <div class="chart">{self.create_radar_chart(list(self.overall["athlete"].keys()), [sum(x["mentions"].values()) for x in self.overall["athlete"].values()], "Interest")}</div>
                    <div class="chart">{self.create_bar_chart(list(self.overall["athlete"].keys()), [round(sum(x["sentiment"])*10, 2) for x in self.overall["athlete"].values()], "Sentiment")}</div>
                </div>

                <h2>Music info</h2>
                <h4>Genres</h4>
                <div class="row">
                    <div class="chart">{self.create_radar_chart(list(self.overall["genre"].keys()), [sum(x["mentions"].values()) for x in self.overall["genre"].values()], "Interest")}</div>
                    <div class="chart">{self.create_bar_chart(list(self.overall["genre"].keys()), [round(sum(x["sentiment"])*10, 2) for x in self.overall["genre"].values()], "Sentiment")}</div>
                </div>
                <div class="row">
                    {" ".join([f'<div class="chart">{self.create_pie_chart(list(data.keys()), list(data.values()), genre)}</div>' for genre,data in self.music_countries.items()])}
                </div>
                <h4>Artists</h4>
                <div class="row">
                    <div class="chart">{self.create_radar_chart(list(self.overall["artist"].keys()), [sum(x["mentions"].values()) for x in self.overall["artist"].values()], "Interest")}</div>
                    <div class="chart">{self.create_bar_chart(list(self.overall["artist"].keys()), [round(sum(x["sentiment"])*10, 2) for x in self.overall["artist"].values()], "Sentiment")}</div>
                </div>
                <h2>Politics info</h2>
                <div class="row">
                    <div class="chart">{self.create_grid_chart([(compass_x/max_r*10, compass_y/max_r*10, '', 'black')])}</div>
                    <div class="chart">{self.create_radar_chart([x for x, y in self.overall["politics"].items()], [x["sentiment"] for x in self.overall["politics"].values()], "Ideologies")}</div>
                    <p>Countries of interest: {",   ".join((self.interest_sum.get("politics", {}) or {}).get("countries", []) or [])}</p>
                </div>
                <h2>Other topics</h2>
                <div class="row">
                    <div class="chart">{self.create_pie_chart([x["interest"] for x in self.others], [x["counter"] for x in self.others], "Other topics")}</div> 
                </div>
            </div>
        </body>
        </html>
        '''

        if do_html:
            with open(self.filename, "w", encoding="utf-8") as file:
                file.write(html_content)
            print(f"HTML file '{self.filename}' has been created.")
             
        
            
        
        
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



profile
OVERALL
/    hashtagy
/    mentions
/    lokacia
/    jazyk  kolac
/    aktivita                       v profile
/    cas dna -> miera aktivity      v profile
/    topic interaction   kolac
/    topic expression    kolac
/    topic interest      kolac
/    topic overall       kolac
    
    
SPORT:
    mentions: pocet tweetov + counter following   ->  interest
    sentiment: iba sport ako taky                 ->  sentiment
    top krajiny: kolac graph                      ->  top krajiny

KLUBY:
    mentions:
    sentiment:

ATLETI:
    mentions:    interest
    sentiment:   sentiment

ZANRE:
    mentions: pocet tweetov + counter following   ->  interest
    sentiment: iba zaner ako taky                 ->  sentiment
    top krajiny: kolac graph                      ->  top krajiny

INTERPRETI:
    mentions:    interest
    sentiment:   sentiment

POLITIKA:


pre kazdy profil
zoznam zaujmov v kazdom poli, miera zaujmu v danych poliach, krajiny zaujmov v danych poliach, lokacia, zamestnanie, hashtagy
casove useky: vyvoj v case, aktivita za interval



BUBBLE
    Zvyraznit ludi co splnaju podmienku nejaku   (mozno console app, ktora vybuduje bublinu a umozni interagovanie s nou)
    napr. miera liberalnosti alebo take volaco, fanuskovia slovanu ...



    INTEREST
        najfollowovanejsi profil clenmi bubliny (list)
        najsledovanejsia topic (list)
        
    

    INTERACTION
        najreagovanejsie topics | sentiment reakcii

    EXPRESSION

    OVERALL
        najoblubenejsi/najneznasanejsi klub
        -//- sport
        -//- atlet
        -//- zaner
        -//- hudobnik
        
        
    
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
        self.avg_activity = None
        self.daily_activity = {i: [] for i in range(24)}
        self.hashtags = {}
        self.summaries = {}
        

        
        for status_id, username, text, type, source_tweet, source_username, hashtags, mentions, created in SCRAPPER.get_tweets(username):
            #print(type, source_tweet)
##            if status_id == "1877127950253822340":
##                print(self.username, status_id, username, text, type, source_tweet, source_username, hashtags, mentions, "\na\na\na\na\na\na\na\na\na\na\na\n")
            tweet = Tweet(status_id, username, text, type, created, source_tweet, source_username, hashtags, mentions)
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
                    self.all_mentions[mention] = [0, []]
                self.all_mentions[mention][0] += 1
                
            for hashtag in hashtags:
                if hashtag not in self.hashtags:
                    self.hashtags[hashtag] = [0, []]
                self.hashtags[hashtag][0] += 1

            self.daily_activity[tweet.time.hour].append(tweet)
        


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

    def summary(self, step=None):

        ### rozdelit si sumar a analyzu na jednoducho vydedukovatelne a tazko
        ### napr. jednoducho = sleduje vela americkych novinarov -> zaujima sa a o americku politiku
##        print(self.username)
##        print(self.expression_sum())
##        print(self.interaction_sum())
##        print(self.interest_sum("profile_analysis.json"))
        all_entries = self.tweets+self.reposts+self.comments+self.quotes
        all_entries.sort(key=lambda x: x.created, reverse=True)
        if all_entries:
            self.avg_activity = f"{len(all_entries)/(((datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + relativedelta(months=1)).replace(day=1) - all_entries[-1].created.replace(day=1)).days / 30)} per month"
        else:
            self.avg_activity = "No activity"
        #print("entries", all_entries)

        if step and all_entries:
            
            start_date = all_entries[-1].created.replace(day=1)
            current_date = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + relativedelta(months=1)).replace(day=1)
            while current_date >= start_date:
                previous_date = current_date - relativedelta(months=step)
                
                if previous_date < start_date:
                    previous_date = start_date

                mentions = {}
                hashtags = {}
                daily = {i: [] for i in range(24)}
                for name, mention in self.all_mentions.items():
                    for date in mention[1]:
                        if date >= previous_date and date < current_date:
                            if name not in mentions:
                                mentions[name] = 0
                            mentions[name] += 1
                            
                for name, hashtag in self.hashtags.items():
                    for date in hashtag[1]:
                        if date >= previous_date and date < current_date:
                            if name not in hashtags:
                                hashtags[name] = 0
                            hashtags[name] += 1

                #print(self.daily_activity)
                number_of_tweets_in_step = 0
                for hour, tweets in self.daily_activity.items():
                    for tweet in tweets:
                        if tweet.created >= previous_date and tweet.created < current_date:
                            daily[hour].append(tweet)
                            number_of_tweets_in_step += 1
                avg_activity = f"{number_of_tweets_in_step/step} per month"
                
                #print(f"{previous_date.strftime('%Y-%m-%d')} -> {current_date.strftime('%Y-%m-%d')}")
                self.summaries[f"{current_date.strftime('%Y-%m-%d')}"] = Summary(self.username, self.expression_sum((previous_date, current_date)), self.interaction_sum((previous_date, current_date)), self.interest_sum("profile_analysis.json"), hashtags, mentions, self.location, avg_activity, daily, (previous_date, current_date))
                
                if previous_date == start_date:
                    break 
                current_date = previous_date
                
        else:
            #print(self.all_mentions)
            min_date = datetime.min
            if all_entries:
                min_date = all_entries[-1].created.replace(day=1)
            interval = (min_date, datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + relativedelta(months=1)).replace(day=1))
            self.summaries[f"{interval[1].strftime('%Y-%m-%d')}"] =  Summary(self.username, self.expression_sum(), self.interaction_sum(), self.interest_sum("profile_analysis.json"), self.hashtags, self.all_mentions, self.location, self.avg_activity, self.daily_activity, interval)
        

        for i in self.summaries.values():
            #print(i)
            i.show()
        #print(self.summaries)
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
            if time_interval:
                if time_interval[0] > tweet.created or time_interval[1] <= tweet.created:
                    continue
                
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
            if time_interval:
                if time_interval[0] > reaction.created or time_interval[1] <= reaction.created:
                    continue
                
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

                
            if type == "politics" or source_tweet_content["type"] == "politics":
                if type == "politics":
                    interaction_summary["politics"]["type"].append(reaction.content["reaction"]["politics"])
                else:
                    sentiment = reaction.content["reaction_sentiment"].lower()
                    add = ""
                    if "disagree" in sentiment:
                        add = "--"
                    elif "neutral" in sentiment:
                        add = "-"
                    elif "agree" in sentiment:
                        add = ""
                    else:
                        raise "INVALID FORMAT OF REACTION SENTIMENT"
                    interaction_summary["politics"]["type reaction"].append(add + source_tweet_content["type"])
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
    def __init__(self, status_id, username, text, type, created, source_tweet=None, source_username=None, hashtags=[], mentions=[]):
        self.status_id = status_id
        self.username = username
        self.text = text
        self.type = type
        self.source_tweet = source_tweet
        self.source_username = source_username
        self.hashtags = hashtags
        self.mentions = mentions
        self.content = None
        temp = created.split()
        temp1 = temp[3].split(":")
        self.created = datetime(int(temp[-1]), int(MONTH_MAP[temp[1]]), int(temp[2]))
        self.time = time(int(temp1[0]), int(temp1[1]), int(temp1[2]))

    def analyse(self):     ###   analyza vynechava niektore,  mozno je to ze v threadoch
        ##########    riesenie => ak najde tweet bez contentu, retrospektivne ho analyzuje
        with open("tweet_analysis.json", 'r', encoding="utf-8") as file:
            cached_tweets = json.load(file)
  
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
                 
        elif self.type == "comment":
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
            DATE: {self.created}
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
                        e.weight["mentions"][direction] += quantity[0]
                            
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
                                    e.weight["mentions"][e.direction(n, self.nodes[i])] += n.profile.all_mentions[i][0]

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

    #step => number of months in one summary
    def profiles_summary(self, step=None):
        for username, node in self.nodes.items():
            node.profile.summary(step)
            for i in node.profile.summaries.values():
                print(i.get_summary())

    def bubble_summary(self):
        return

            

        




            
        

''' hrany s interakciami ohladom konkretnej temy '''



SB = SocialBubble("pushkicknadusu", "profile_centered", depth=0)

SB.create_graph()

SB.visualize_graph()

##opd = SB.get_outside_profiles_data(THRESHOLD)

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

##SB.profile_analysis(opd)

SB.tweet_analysis()

SB.profiles_summary()


'''
1. VYTVORIT KVALITNU ANALYZU PROFILU + VYOBRAZENIE


mozne funkcie:
    UNION, INTERSECTION
    AK JE NEJAKA TOPIC, ZVYRAZNIT TIE KTORYCH SA TO TYKA + HRANY (INTERAKCIA, ROVNAKY FOLLOW NA DANU TOPIC ...)

    CONTAINS => prejdenie tweetov s tým, že sa analyzuje, či obsahuje danú tému a sentiment na ňu

'''
