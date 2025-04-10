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
from pyvis.network import Network
import community as community_louvain
import random


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
AI_GENERALISATION_PARSER = AIAnalysis.GPT4o()


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
        
    def compute_interaction_strength(self, PROFILES=[]):
        total = 0
        for edge in self.edges:
            if PROFILES and edge.get_second_node(self).profile.username not in PROFILES:
                continue
            total += edge.get_weight_eval()
        return total

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
    def get_weight_eval(self):
        reactions_1_to_2 = len(self.weight["reactions"]["1->2"])
        reactions_2_to_1 = len(self.weight["reactions"]["2->1"])
        reaction_base = (reactions_1_to_2 + reactions_2_to_1) * 1  # Base: 1 per reaction
        reaction_bonus = 2 if (reactions_1_to_2 > 0 and reactions_2_to_1 > 0) else 0  # Mutual bonus

        # Mentions
        mentions_1_to_2 = self.weight["mentions"]["1->2"]
        mentions_2_to_1 = self.weight["mentions"]["2->1"]
        mention_base = (mentions_1_to_2 + mentions_2_to_1) * 0.5  # Base: 0.5 per mention
        mention_bonus = 1 if (mentions_1_to_2 > 0 and mentions_2_to_1 > 0) else 0  # Mutual bonus

        # Follows
        follows = self.weight["follows"]
        follow_bonus = 3 if follows == "friends" else 1 if follows in ["->", "<-"] else 0

        # Total weight
        total_weight = reaction_base + reaction_bonus + mention_base + mention_bonus + follow_bonus
        return total_weight

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

    def __contains__(self, item):
        if isinstance(item, Node):
            return self.node1 == item or self.node2 == item


        
class Summary:
    def __init__(self, username, expression, interaction, interest, hashtags, mentions, location, avg_activity, daily_activity, interval):
        self.username = username
        self.expression_sum = expression
        self.interaction_sum = interaction
        self.interest_sum = interest
        self.overall = self.overall_sum()
        self.hashtags = hashtags
        self.mentions = mentions
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
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.5)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + data["c/p tweets"] + len(data["reaction sentiments"])
            temp = process.extractOne(sport, self.sports.keys()) or [0,0]
            if sport not in self.sports and temp[1] < 85:
                self.sports[sport.lower()] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": mentions, "interest": 0}}
            else:
                self.sports[temp[0]]["sentiment"].append(sentiment)
                self.sports[temp[0]]["sentiment"].append(reaction_sentiment)
                self.sports[temp[0]]["mentions"]["interaction"] += mentions

        if self.interest_sum.get("sport", None):
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
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.7)
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
            
                

        if self.interest_sum.get("sport", None):
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
            for sport in [data["sport"]]:
                if sport.lower() not in self.sport_countries:
                    self.sport_countries[sport.lower()] = {}
                if data["country"].lower() not in self.sport_countries[sport.lower()]:
                    self.sport_countries[sport.lower()][data["country"].lower()] = 0
                self.sport_countries[sport.lower()][data["country"].lower()] += 1
            

        for player, data in self.interaction_sum["players"].items():
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.7)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + len(data["reaction sentiments"])
            temp = process.extractOne(sport, self.sports.keys()) or [0,0]
            if player not in self.players and temp[1] < 85:
                self.athletes[player] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": mentions, "interest": 0}}
            else:
                self.athletes[temp[0]]["sentiment"].append(sentiment)
                self.athletes[temp[0]]["sentiment"].append(reaction_sentiment)
                self.athletes[temp[0]]["mentions"]["interaction"] += mentions
                
            for sport in [data["sport"]]:
                if sport.lower() not in self.sport_countries:
                    self.sport_countries[sport.lower()] = {}
                if data["country"].lower() not in self.sport_countries[sport.lower()]:
                    self.sport_countries[sport.lower()][data["country"].lower()] = 0
                self.sport_countries[sport.lower()][data["country"].lower()] += 1
            

        if self.interest_sum.get("sport", None):
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
                                        self.athletes[temp[0]]["sentiment"].append(sentiment)
                                        self.athletes[temp[0]]["mentions"]["interest"] += mentions
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
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.5)
            reaction_sentiment = self.interpret_sentiment_list(data["reaction sentiments"], 0.8, 0.1, -0.5)
            mentions = len(data["sentiments"]) + data["artist tweets"] + len(data["reaction sentiments"])
            temp = process.extractOne(genre, self.genres.keys()) or [0,0]
            if genre not in self.genres and temp[1] < 85:
                self.genres[genre.lower()] = {"sentiment": [sentiment], "mentions": {"expression": 0, "interaction": mentions, "interest": 0}}
            else:
                self.genres[temp[0]]["sentiment"].append(sentiment)
                self.genres[temp[0]]["sentiment"].append(reaction_sentiment)
                self.genres[temp[0]]["mentions"]["interaction"] += mentions

        if self.interest_sum.get("music", None):
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
            sentiment = self.interpret_sentiment_list(data["sentiments"], 1.1, 0.4, -0.7)
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
            

        if self.interest_sum.get("music", None):
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

                        
        self.others = copy.deepcopy(self.interest_sum.get("other_interests", []))

        
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

        if self.interest_sum.get("politics", None):
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

    def generate_html(self, do_html=False):
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
        if not max_r:
            max_r = float('inf')
        if do_html:
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
                        <p>Aktivita: {self.avg_activity} per month</p>
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
                    self.all_mentions[mention] = []
                self.all_mentions[mention].append(tweet.created)
                
            for hashtag in hashtags:
                if hashtag not in self.hashtags:
                    self.hashtags[hashtag] = []
                self.hashtags[hashtag].append(tweet.created)

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

    def summary(self, step=1):

        ### rozdelit si sumar a analyzu na jednoducho vydedukovatelne a tazko
        ### napr. jednoducho = sleduje vela americkych novinarov -> zaujima sa a o americku politiku
##        print(self.username)
##        print(self.expression_sum())
##        print(self.interaction_sum())
##        print(self.interest_sum("profile_analysis.json"))
        all_entries = self.tweets+self.reposts+self.comments+self.quotes
        all_entries.sort(key=lambda x: x.created, reverse=True)
        if all_entries:
            self.avg_activity = len(all_entries)/(((datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + relativedelta(months=1)).replace(day=1) - all_entries[-1].created.replace(day=1)).days / 30)
        else:
            self.avg_activity = 0
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
                for name, dates in self.all_mentions.items():
                    for date in dates:
                        if date >= previous_date and date < current_date:
                            if name not in mentions:
                                mentions[name] = 0
                            mentions[name] += 1
                            
                for name, dates in self.hashtags.items():
                    for date in dates:
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
                avg_activity = number_of_tweets_in_step/step
                
                #print(f"{previous_date.strftime('%Y-%m-%d')} -> {current_date.strftime('%Y-%m-%d')}")
                self.summaries[current_date] = Summary(self.username, self.expression_sum((previous_date, current_date)), self.interaction_sum((previous_date, current_date)), self.interest_sum("profile_analysis.json"), hashtags, mentions, self.location, avg_activity, daily, (previous_date, current_date))
                
                if previous_date == start_date:
                    break 
                current_date = previous_date
                
        else:
            #print(self.all_mentions)
            min_date = datetime.min
            if all_entries:
                min_date = all_entries[-1].created.replace(day=1)
            interval = (min_date, (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + relativedelta(months=1)).replace(day=1))
            self.summaries[interval[1]] =  Summary(self.username, self.expression_sum(), self.interaction_sum(), self.interest_sum("profile_analysis.json"), self.hashtags, self.all_mentions, self.location, self.avg_activity, self.daily_activity, interval)
        

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
                print("ERROR expression tweet.content[\"type\"]")
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
                print("\nERROR: type = reaction.content[\"reaction\"][\"type\"]  ")
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
                print("\nZMAZANY OG TWEET \n")
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

        return PROFILE_AI_ANALYSER.profiles_summary(data) if data else {}
        

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
        self.decentralised_profiles = args.get("profiles", None)
        self.type = type
        self.minimal_edge_weight = args.get("minimal_edge_weight", 0)
        if self.type == "entity_centered":
            self.depth = 1
        else:
            self.depth = args["depth"]
        self.username = username
        self.edges = []
        self.nodes = {}  ### name : Node
        self.followed_outside_bubble = {}
        self.following_outside_bubble = {}
        self.mentioned_outside_bubble = {}
        self.interacted_outside_bubble = {}
        

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
                        e = self.exist_edge(node, n)
                        if not e:
                            e = Edge("friends", node, n)
                            self.edges.append(e)
                        new_queue.append(n)
                        self.nodes[friend] = n
                        if e not in n.edges:
                            n.edges.append(e)
                        if e not in node.edges:
                            node.edges.append(e)
                        
                    visited.append(node.profile.username)
                    
                queue = new_queue

            ### doplni chybajuce friendship edges
            for node in queue:
                for friend in node.profile.friends:
                    if friend in self.nodes.keys():
                        e = self.exist_edge(node, self.nodes[friend])
                        if not e:
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
                            e = Edge("->", node, self.nodes[following])
                            self.edges.append(e)
                            node.edges.append(e)
                            self.nodes[following].edges.append(e)
                            
                    else:
                        ### pridat following do struktury kde sa spocita kym je sledovana
                        if following not in self.followed_outside_bubble:
                            self.followed_outside_bubble[following] = []
                        self.followed_outside_bubble[following].append(name)
                        
                for follower in list(set(node.profile.followers) - set(node.profile.friends)):
                    if follower in self.nodes.keys():
                        if not self.exist_edge(node, self.nodes[follower]):
                            e = Edge("->", node, self.nodes[follower])
                            self.edges.append(e)
                            node.edges.append(e)
                            self.nodes[follower].edges.append(e)
                    else:
                        ### pridat following do struktury kde sa spocita kym je sledovana
                        if follower not in self.following_outside_bubble:
                            self.following_outside_bubble[follower] = []
                        self.following_outside_bubble[follower].append(name)

            for name in list(set(self.following_outside_bubble.keys()) & set(self.followed_outside_bubble.keys())):
                self.nodes[name] = Node(name)
                for name2 in self.following_outside_bubble[name]:
                    e = self.exist_edge(self.nodes[name], self.nodes[name2])
                    if not e:
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
                            if self.nodes[mention] == node:
                                continue
                            e = Edge("X", self.nodes[mention], node)
                            self.edges.append(e)
                            node.edges.append(e)
                            self.nodes[mention].edges.append(e)
                        direction = e.direction(self.nodes[mention], node)
                        e.weight["mentions"][direction] += len(quantity)
                            
                    else:
                        if mention not in self.mentioned_outside_bubble:
                            self.mentioned_outside_bubble[mention] = []
                        self.mentioned_outside_bubble[mention].append(username)

                   
            ### spracovat reakcie tweetove
            queue = list(self.nodes.values())
            self.interacted_outside_bubble = {}
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
                                if node == node2:
                                    continue
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
                                    if n == node:
                                        continue
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
                                        if n == self.nodes[i]:
                                            continue
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
                                        if n == self.nodes[i]:
                                            continue
                                        e = Edge("X", n, self.nodes[i])
                                        n.edges.append(e)
                                        self.nodes[i].edges.append(e)
                                        self.edges.append(e)
                                    e.weight["mentions"][e.direction(n, self.nodes[i])] += len(n.profile.all_mentions[i])

                                for i in self.mentioned_outside_bubble.get(tweet.source_username, []):
                                    temp = self.exist_edge(n, self.nodes[i])
                                    if temp:
                                        e = temp
                                    else:
                                        if n == self.nodes[i]:
                                            continue
                                        e = Edge("X", n, self.nodes[i])
                                        n.edges.append(e)
                                        self.nodes[i].edges.append(e)
                                        self.edges.append(e)
                                    e.weight["mentions"][e.direction(self.nodes[i], n)] += len(self.nodes[i].all_mentions[tweet.source_username])




                                ''' prekontrolovat ci followers/ing dole'''




                                for i in list(set(n.profile.followers) & set(self.nodes.keys())):
                                    temp = self.exist_edge(n, self.nodes[i])
                                    if temp:
                                        e = temp
                                    else:
                                        if n == self.nodes[i]:
                                            continue
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
                                if tweet.source_username not in self.interacted_outside_bubble:
                                    self.interacted_outside_bubble[tweet.source_username] = []
                                self.interacted_outside_bubble[tweet.source_username].append(tweet.username)
                        
            #print(self.followed_outside_bubble, self.mentioned_outside_bubble)
                
            ## pridat interagovane profily mimo bubliny do nejakeho zoznamu, zoradene podla poctu interakcii clenov bubliny a followov


            ### ak je repost/comment/quote a og tweet autor sleduje niekoho v bubline tak je pridany


        
        elif self.type == "decentralised":
            tuto som skoncil
            '''
                Vzajomne interakcie
                Pre kazdy profil spolocne sledovane/sledovany

                1.  prejde vsetky a vytvori profil, pre kazdy vytvori do hlbky jedna prieskum
                ak ich spaja nejaky friend tak ho to prida
            '''
            for username in self.decentralised_profiles:
                self.nodes[username] = Node(username)

            nodes_to_add = {}
            for node1 in self.nodes.values():
                for node2 in self.nodes.values():
                    if node1 == node2:
                        continue
                    intersection = set(node1.profile.friends) & set(node2.profile.friends)
                    if not intersection:
                        intersection = set(node1.profile.friends) & set(node2.profile.followers)
                    if not intersection:
                        intersection = set(node1.profile.friends) & set(node2.profile.following)
                    for mutual_friend in intersection:
                        if mutual_friend not in nodes_to_add:
                            nodes_to_add[mutual_friend] = Node(mutual_friend)

            for username, node in nodes_to_add.items():
                self.nodes[username] = node

            for node1 in self.nodes.values():
                for node2 in self.nodes.values():
                    if self.exist_edge(node1, node2) or node1 == node2:
                        continue
                    if node1.profile.username in node2.profile.friends:
                        self.edges.append(Edge("friends", node1, node2))
                    elif node1.profile.username in node2.profile.followers:
                        self.edges.append(Edge("->", node1, node2))
                    elif node1.profile.username in node2.profile.following:
                        self.edges.append(Edge("<-", node1, node2))

            for username, node in self.nodes.items():
                for followed in list(set(node.profile.following)-set(self.nodes.keys())):
                    if followed not in self.followed_outside_bubble:
                        self.followed_outside_bubble[followed] = []
                    self.followed_outside_bubble[followed].append(username)
                    
                for following in list(set(node.profile.followers)-set(self.nodes.keys())):
                    if following not in self.following_outside_bubble:
                        self.following_outside_bubble[following] = []
                    self.following_outside_bubble[following].append(username)
                
               
        
                        
            'popridavat chybajuce edges az na konci naraz'
            ### najdenie vzajomnych prepojeni




        ### prejde follows ludi v bubline, hlada bublina & (follows - friends), spocita co je kym followovane pri profiloch mimo bubliny
        ### prejde tweety, pozrie ci su nejake retweety


    def visualize_graph(self):        
        # Create a directed network
        net = Network(directed=True, notebook=False, height="750px", width="100%")
        
        # Add nodes
        for node in self.nodes.values():
            net.add_node(node.profile.username, label=node.profile.username)
        
        # Define edge lists for different colors
        edge_colors = {
            "->": "red",
            "<-": "red",
            "friends": "green",
            "X": "blue"
        }
        
        # Add edges with color classification
        for edge in self.edges:
            node1 = list(edge.directions.keys())[0]
            node2 = edge.get_second_node(node1)
            follow_type = edge.weight.get("follows", "friends")

            if follow_type == "->":
                net.add_edge(node1.profile.username, node2.profile.username, color="red", arrows="to")
            elif follow_type == "<-":
                net.add_edge(node2.profile.username, node1.profile.username, color="red", arrows="to")
            elif follow_type == "friends":
                # Bidirectional edge (arrows on both ends)
                net.add_edge(node1.profile.username, node2.profile.username, color="green", arrows="to, from")
            elif follow_type == "X":
                net.add_edge(node1.profile.username, node2.profile.username, color="blue", arrows="to")
        # Configure physics for better layout
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        """)
        
        # Show the network
        net.show("social_graph.html", notebook=False)

    def visualize_outside_relations(self):
        # Clean up the dictionaries first
        for i in self.nodes.keys():
            if i in self.followed_outside_bubble:
                self.followed_outside_bubble.pop(i)
            if i in self.following_outside_bubble:
                self.following_outside_bubble.pop(i)
            if i in self.mentioned_outside_bubble:
                self.mentioned_outside_bubble.pop(i)
            if i in self.interacted_outside_bubble:
                self.interacted_outside_bubble.pop(i)
        
        # Create a new Pyvis network
        net = Network(height="1000px", width="100%", bgcolor="#222222", font_color="white", directed=True)
        
        # Convert all node IDs to strings
        def safe_str(id):
            return str(id) if id is not None else "None"
        
        # Get all unique profiles and users
        followed_profiles = set(safe_str(p) for p in self.followed_outside_bubble.keys()) - set(safe_str(n) for n in self.nodes.keys())
        following_profiles = set(safe_str(p) for p in self.following_outside_bubble.keys()) - set(safe_str(n) for n in self.nodes.keys())
        
        # Add all nodes with their properties
        # Following profiles (left group) - blue boxes
        for profile in following_profiles:
            net.add_node(profile, 
                        label=profile,
                        color="#45B7D1",
                        shape="box",
                        font={'size': 14},
                        margin=10)
        
        # Main users (center group) - turquoise circles
        for user in [safe_str(n) for n in self.nodes.keys()]:
            net.add_node(user,
                        label=user,
                        color="#4ECDC4",
                        font={'size': 12},
                        margin=8)
        
        # Followed profiles (right group)
        for profile in followed_profiles:
            # Check if profile is in mentioned or interacted
            if profile in [safe_str(p) for p in self.mentioned_outside_bubble.keys()] or \
               profile in [safe_str(p) for p in self.interacted_outside_bubble.keys()]:
                net.add_node(profile,
                            label=profile,
                            color="#FFA500",  # Orange for mentioned/interacted
                            shape="box",
                            font={'size': 14},
                            margin=10)
            else:
                net.add_node(profile,
                            label=profile,
                            color="#FF6B6B",  # Red for regular followed
                            shape="box",
                            font={'size': 14},
                            margin=10)
        
        # Add nodes from mentioned/interacted that aren't in followed
        all_profiles = followed_profiles.union(following_profiles)
        mentioned_interacted = set(safe_str(p) for p in self.mentioned_outside_bubble.keys()).union(
                              set(safe_str(p) for p in self.interacted_outside_bubble.keys()))
        
        for profile in mentioned_interacted:
            if profile not in all_profiles and profile not in [safe_str(n) for n in self.nodes.keys()]:
                net.add_node(profile,
                            label=profile,
                            color="#FFA500",
                            shape="box",
                            font={'size': 14},
                            margin=10)
        
        # Add edges - following relationships (always blue)
        for profile, users in self.following_outside_bubble.items():
            profile_str = safe_str(profile)
            for user in users:
                user_str = safe_str(user)
                if user_str in [safe_str(n) for n in self.nodes.keys()] and profile_str in following_profiles:
                    net.add_edge(profile_str, user_str, 
                               color="#45B7D1",  # Blue for following
                               width=1.5,
                               arrows='to')
        
        # Add edges - followed relationships (red unless from mentioned/interacted)
        for profile, users in self.followed_outside_bubble.items():
            profile_str = safe_str(profile)
            for user in users:
                user_str = safe_str(user)
                if user_str in [safe_str(n) for n in self.nodes.keys()] and profile_str in followed_profiles:
                    # Check if this edge comes from a mentioned/interacted relationship
                    is_mentioned = any(user_str == safe_str(u) and profile_str == safe_str(p) 
                                     for p, us in self.mentioned_outside_bubble.items() for u in us)
                    is_interacted = any(user_str == safe_str(u) and profile_str == safe_str(p) 
                                      for p, us in self.interacted_outside_bubble.items() for u in us)
                    
                    if is_mentioned or is_interacted:
                        net.add_edge(user_str, profile_str,
                                   color="#FFA500",  # Orange for mentioned/interacted
                                   width=1.5,
                                   arrows='to')
                    else:
                        net.add_edge(user_str, profile_str,
                                   color="#FF6B6B",  # Red for regular followed
                                   width=1.5,
                                   arrows='to')
        
        # Add edges for mentioned/interacted that aren't in followed
        for profile, users in self.mentioned_outside_bubble.items():
            profile_str = safe_str(profile)
            for user in users:
                user_str = safe_str(user)
                if user_str in [safe_str(n) for n in self.nodes.keys()]:
                    # Only add if not already in followed relationships
                    if not any(user_str == safe_str(u) and profile_str == safe_str(p) 
                              for p, us in self.followed_outside_bubble.items() for u in us):
                        net.add_edge(user_str, profile_str,
                                   color="#FFA500",
                                   width=1.5,
                                   arrows='to')
        
        for profile, users in self.interacted_outside_bubble.items():
            profile_str = safe_str(profile)
            for user in users:
                user_str = safe_str(user)
                if user_str in [safe_str(n) for n in self.nodes.keys()]:
                    # Only add if not already in followed relationships
                    if not any(user_str == safe_str(u) and profile_str == safe_str(p) 
                              for p, us in self.followed_outside_bubble.items() for u in us):
                        net.add_edge(user_str, profile_str,
                                   color="#FFA500",
                                   width=1.5,
                                   arrows='to')
        
        # Configure physics
        net.toggle_physics(True)
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        """)
        
        # Save and show the graph
        net.show("bubble_graph.html", notebook=False)
        return net


    def visualize_hashtags(self, start=datetime.min, end=datetime.now()):
        """
        Visualize hashtag usage across all profiles in the network.
        
        Args:
            start (datetime): Start date for filtering hashtags
            end (datetime): End date for filtering hashtags
        """
        # Create a new Pyvis network
        net = Network(height="1000px", width="100%", bgcolor="#222222", font_color="white", directed=True)
        
        # Add all profile nodes (left side)
        profile_nodes = {}
        for i, profile_id in enumerate(self.nodes.keys()):
            profile = self.nodes[profile_id]
            net.add_node(f"profile_{profile_id}",
                        color="#4ECDC4",
                        font={'size': 12},
                        margin=8)  # Profiles on level 1 (left)
            profile_nodes[profile_id] = f"profile_{profile_id}"
        
        # Collect all hashtags used in date range
        hashtag_counts = {}
        for profile_id in self.nodes.keys():
            profile = self.nodes[profile_id]
            for hashtag, dates in profile.profile.hashtags.items():
                for date in dates:
                    if date >= start and date <= end:
                        hashtag_counts[hashtag] = hashtag_counts.get(hashtag, 0) + 1
        
        # Add hashtag nodes (right side) - size based on usage frequency
        hashtag_nodes = {}
        for i, (hashtag, count) in enumerate(sorted(hashtag_counts.items(), key=lambda x: -x[1])):
            size = 10 + min(count, 20)  # Cap size at 30
            net.add_node(f"hashtag_{hashtag}", 
                        label=f"#{hashtag}",
                        color="#45B7D1",
                        shape="box",
                        font={'size': 14},
                        margin=10
                        )  # Hashtags on level 2 (right)
            hashtag_nodes[hashtag] = f"hashtag_{hashtag}"
        
        # Add edges between profiles and hashtags they've used
        for profile_id in self.nodes.keys():
            profile = self.nodes[profile_id]
            for hashtag, dates in profile.profile.hashtags.items():
                if hashtag in hashtag_nodes:  # Only if hashtag was used in date range
                    usage_count = sum(1 for date in dates if date >= start and date <= end)
                    if usage_count > 0:
                        net.add_edge(profile_nodes[profile_id], 
                                   hashtag_nodes[hashtag],
                                   value=usage_count,
                                   color="#FFA500",
                                   width=1.5,
                                   arrows='to')  # Thicker for more usage
        
        # Configure physics for better layout
        net.toggle_physics(True)
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        """)
        

        
        # Add date range to title
        title = f"Hashtag Usage: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        
        # Save and show the graph
        filename = f"hashtag_network_{start.strftime('%Y%m%d')}_to_{end.strftime('%Y%m%d')}.html"
        net.show(filename, notebook=False)
        return net
        

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
    def profiles_summary(self, step=1):
        self.all_sums = {}
        for username, node in self.nodes.items():
            print(username)
            node.profile.summary(step)

            for date, summary in node.profile.summaries.items():
                print(date)
                if date not in self.all_sums:
                    self.all_sums[date] = []
                self.all_sums[date].append(summary)
                
        return self.all_sums


class BubbleSummary:
    def __init__(self, all_sums, social_bubble):
        self.all_sums = all_sums
        self.social_bubble = social_bubble
        self.evolution_stats = {
            "languages": {},
            "daily_activity": {},
            "top_hashtags": {}, # vezme top hashtagy z kazdeho profilu pre kazdy mesiac
            "top_mentions": {},
            "avg_activity": {},
            "daily_activity": {},
            
            "hashtags": {},
            "mentions": {},

            "ideologies": {},   # expressed ideologies
            
            
            
        }
        '''   stat: profil: interval: value   '''

        self.overall_stats = {
##            "mention_usage": {},
##            "most_followed_profile": {},
            "locations": {},
            "topic_spread": {},
            "other_topics": {},  # topic : {profiles : counter}
            "compass": {},   # profile : compass
             # ideology : {+ : [profiles], - : [profiles]}
            "ideologies_overall": {}
            
# overall
            
            
            
        }
######        self.bubble_summary()

    def create_pie_chart(self, labels, values, title):
        if labels and values:
            pie_chart = go.Figure(go.Pie(labels=labels, values=values, title=title))
            pie_chart.update_layout(width=550, height=550)
            return pie_chart.to_html(full_html=False)

    def create_radar_chart(self, labels, values, title):
        if labels and values:
            radar_chart = go.Figure()
            radar_chart.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself'))
            radar_chart.update_layout(title=title, polar=dict(radialaxis=dict(visible=False)), width=550, height=550)
            return radar_chart.to_html(full_html=False)

    def create_bar_chart(self, labels, values, title):
        if labels and values:
            bar_chart = go.Figure(go.Bar(x=labels, y=values, marker_color=['red' if v < -2 else 'yellow' if -2 <= v <= 2 else 'green' for v in values]))
            bar_chart.update_layout(title=title, yaxis=dict(range=[-10, 10]), width=550, height=550)
            return bar_chart.to_html(full_html=False)

    def create_line_graph(self, x_values, y_values_list, line_names, title):
        if not x_values or not y_values_list or not line_names:
            return None
            
        if len(y_values_list) != len(line_names):
            return None
            
        line_graph = go.Figure()
        
        for y_values, name in zip(y_values_list, line_names):
            line_graph.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                name=name,
                mode='lines+markers'
            ))
        
        line_graph.update_layout(
            title=title,
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            width=550,
            height=550,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return line_graph.to_html(full_html=False)

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

        
    def bubble_summary(self):
        for date, summaries in self.all_sums.items():
            for summary in summaries:
                data = summary.get_summary()
                for language, count in data["languages"].items():
                    if date not in self.evolution_stats["languages"]:
                        self.evolution_stats["languages"][date] = {}
                    if language not in self.evolution_stats["languages"][date]:
                        self.evolution_stats["languages"][date][language] = {}
                    if data['username'] not in self.evolution_stats["languages"][date][language]:
                        self.evolution_stats["languages"][date][language][data['username']] = 0          
                    self.evolution_stats["languages"][date][language][data['username']] += count

                for daily_activity, count in data["daily_activity"].items():
                    if date not in self.evolution_stats["daily_activity"]:
                        self.evolution_stats["daily_activity"][date] = {}
                    if daily_activity not in self.evolution_stats["daily_activity"][date]:
                        self.evolution_stats["daily_activity"][date][daily_activity] = {}
                    if data['username'] not in self.evolution_stats["daily_activity"][date][daily_activity]:
                        self.evolution_stats["daily_activity"][date][daily_activity][data['username']] = 0          
                    self.evolution_stats["daily_activity"][date][daily_activity][data['username']] += count

                
                if date not in self.evolution_stats["avg_activity"]:
                    self.evolution_stats["avg_activity"][date] = {}
                if data['username'] not in self.evolution_stats["avg_activity"][date]:
                    self.evolution_stats["avg_activity"][date][data['username']] = 0          
                self.evolution_stats["avg_activity"][date][data['username']] += data["avg_activity"]

                
                if data["location"] not in self.overall_stats["locations"]:
                    self.overall_stats["locations"][data["location"]] = []
                self.overall_stats["locations"][data["location"]].append(data["username"])


                for hashtag, count in summary.hashtags.items():
                    if date not in self.evolution_stats["hashtags"]:
                        self.evolution_stats["hashtags"][date] = {}
                    if hashtag not in self.evolution_stats["hashtags"][date]:
                        self.evolution_stats["hashtags"][date][hashtag] = {}
                    if data['username'] not in self.evolution_stats["hashtags"][date][hashtag]:
                        self.evolution_stats["hashtags"][date][hashtag][data['username']] = 0
                    self.evolution_stats["hashtags"][date][hashtag][data['username']] = count

                for mention, count in summary.mentions.items():
                    if date not in self.evolution_stats["mentions"]:
                        self.evolution_stats["mentions"][date] = {}
                    if mention not in self.evolution_stats["mentions"][date]:
                        self.evolution_stats["mentions"][date][mention] = {}
                    if data['username'] not in self.evolution_stats["mentions"][date][mention]:
                        self.evolution_stats["mentions"][date][mention][data['username']] = 0
                    self.evolution_stats["mentions"][date][mention][data['username']] = count

                for topic in ["sport", "club", "artist", "genre", "athlete"]:
                    for item, item_data in data["overall"][topic].items():
                        if f"{topic}_tweet_sentiment" not in self.evolution_stats:
                            self.evolution_stats[f"{topic}_tweet_sentiment"] = {}
                        if f"{topic}_overall_sentiment" not in self.evolution_stats:
                            self.evolution_stats[f"{topic}_overall_sentiment"] = {}
                        if f"{topic}_tweet_mentions" not in self.evolution_stats:
                            self.evolution_stats[f"{topic}_tweet_mentions"] = {}
                        if f"{topic}_overall_mentions" not in self.evolution_stats:
                            self.evolution_stats[f"{topic}_overall_mentions"] = {}

                        
                        if date not in self.evolution_stats[f"{topic}_tweet_sentiment"]:
                            self.evolution_stats[f"{topic}_tweet_sentiment"][date] = {}
                        if date not in self.evolution_stats[f"{topic}_overall_sentiment"]:
                            self.evolution_stats[f"{topic}_overall_sentiment"][date] = {}
                        if date not in self.evolution_stats[f"{topic}_tweet_mentions"]:
                            self.evolution_stats[f"{topic}_tweet_mentions"][date] = {}
                        if date not in self.evolution_stats[f"{topic}_overall_mentions"]:
                            self.evolution_stats[f"{topic}_overall_mentions"][date] = {}
                            
                        if item not in self.evolution_stats[f"{topic}_tweet_sentiment"][date]:
                            self.evolution_stats[f"{topic}_tweet_sentiment"][date][item] = {}
                        if item not in self.evolution_stats[f"{topic}_overall_sentiment"][date]:
                            self.evolution_stats[f"{topic}_overall_sentiment"][date][item] = {}
                        if item not in self.evolution_stats[f"{topic}_tweet_mentions"][date]:
                            self.evolution_stats[f"{topic}_tweet_mentions"][date][item] = {}
                        if item not in self.evolution_stats[f"{topic}_overall_mentions"][date]:
                            self.evolution_stats[f"{topic}_overall_mentions"][date][item] = {}
                            
                        if data['username'] not in self.evolution_stats[f"{topic}_tweet_sentiment"][date][item]:
                            self.evolution_stats[f"{topic}_tweet_sentiment"][date][item][data['username']] = 0
                        if data['username'] not in self.evolution_stats[f"{topic}_overall_sentiment"][date][item]:
                            self.evolution_stats[f"{topic}_overall_sentiment"][date][item][data['username']] = 0
                        if data['username'] not in self.evolution_stats[f"{topic}_tweet_mentions"][date][item]:
                            self.evolution_stats[f"{topic}_tweet_mentions"][date][item][data['username']] = 0
                        if data['username'] not in self.evolution_stats[f"{topic}_overall_mentions"][date][item]:
                            self.evolution_stats[f"{topic}_overall_mentions"][date][item][data['username']] = 0
                            
                        self.evolution_stats[f"{topic}_tweet_sentiment"][date][item][data['username']] += sum(item_data['sentiment'][:-1] if item_data["mentions"]['interest'] > 0 else item_data['sentiment'])
                        self.evolution_stats[f"{topic}_overall_sentiment"][date][item][data['username']] += sum(item_data['sentiment'])
                        self.evolution_stats[f"{topic}_tweet_mentions"][date][item][data['username']] += item_data['mentions'].get('expression', 0)+item_data['mentions'].get('interaction', 0)
                        self.evolution_stats[f"{topic}_overall_mentions"][date][item][data['username']] += sum(item_data['mentions'].values())
                        

                for topic in data["overall"]["other"] or []:
                    if topic['interest'] not in self.overall_stats['other_topics']:
                        self.overall_stats['other_topics'][topic['interest']] = {}
                    if data['username'] not in self.overall_stats['other_topics'][topic['interest']]:
                        self.overall_stats['other_topics'][topic['interest']][data['username']] = 0
                    self.overall_stats['other_topics'][topic['interest']][data['username']] += topic['counter']
                for i in ["politics", "sport", "music"]:
                    if i not in self.overall_stats['other_topics']:
                        self.overall_stats['other_topics'][i] = {}
                    if data['username'] not in self.overall_stats['other_topics'][i]:
                        self.overall_stats['other_topics'][i][data['username']] = 0
                        
                    self.overall_stats['other_topics'][i][data['username']] += data["topic_counts"][2][i]

                
                if data['username'] not in self.overall_stats['compass']:
                    self.overall_stats['compass'][data['username']] = [0,0,0]
                for i in range(3):
                    self.overall_stats['compass'][data['username']][i] += data['compass'][i]


                for ideology, ideology_data in data["overall"]["politics"].items():
                    if date not in self.evolution_stats["ideologies"]:
                        self.evolution_stats["ideologies"][date] = {}
                    if ideology not in self.overall_stats["ideologies_overall"]:
                        self.overall_stats["ideologies_overall"][ideology] = {}
                    if ideology not in self.evolution_stats["ideologies"][date]:
                        self.evolution_stats["ideologies"][date][ideology] = {}
                    if data['username'] not in self.overall_stats["ideologies_overall"][ideology]:
                        self.overall_stats["ideologies_overall"][ideology][data['username']] = 0
                    if data['username'] not in self.evolution_stats["ideologies"][date][ideology]:
                        self.evolution_stats["ideologies"][date][ideology][data['username']] = 0
                    self.overall_stats["ideologies_overall"][ideology][data['username']] += ideology_data["sentiment"]
                    self.evolution_stats["ideologies"][date][ideology][data['username']] += ideology_data["mentions"]["ex/int"]
 
                        
##        print(self.evolution_stats[f"{topic}_overall_sentiment"], self.evolution_stats[f"{topic}_overall_mentions"])
        for topic in ["sport", "club", "artist", "genre", "athlete"]:
            for i in ["sentiment", "mentions"]:
                result = {}
                for datetime_data in self.evolution_stats[f"{topic}_overall_{i}"].values():
                    for item, profiles in datetime_data.items():
                        for profile, value in profiles.items():
                            if item not in result:
                                result[item] = {}
                            if profile not in result[item]:
                                result[item][profile] = 0
                            result[item][profile] += value
                self.evolution_stats[f"{topic}_overall_{i}"] = result      
                        



                    
                

        with open("bubble_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "evolution": self.convert_datetime_keys_to_strings(self.evolution_stats),
                    "overall": self.convert_datetime_keys_to_strings(self.overall_stats)
                },
                f,
                indent=4,
                ensure_ascii=False
            )            

##                jazyk sum
##                denna akt sum
##                ziskat hashtagy z profiles
##                ziskat mentions z profiles
##            {
##                "username": self.username,
##                "expression_sum": self.expression_sum,
##                "interaction_sum": self.interaction_sum,
##                "interest_sum": self.interest_sum,
##                "overall": self.overall,
##                "top_hashtags": self.top_hashtags,
##                "top_mentions": self.top_mentions,
##                "location": self.location,
##                "avg_activity": self.avg_activity,
##                "daily_activity": self.daily_activity,
##                "interval": self.interval,
##                "languages": self.languages,
##                "compass": self.compass,
##                "topic_counts": self.topic_counts
##            }

    # date lang profile count
    def convert_datetime_keys_to_strings(self, obj):
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                # Convert datetime key to string
                new_key = key.isoformat() if isinstance(key, datetime) else key
                
                # Recursively process the value
                new_value = self.convert_datetime_keys_to_strings(value)
                
                new_dict[new_key] = new_value
            return new_dict
        elif isinstance(obj, list):
            return [self.convert_datetime_keys_to_strings(item) for item in obj]
        else:
            return obj




        
    def avg_activity_evolution(self, PROFILES=None):
        #print(self.evolution_stats["avg_activity"])
        # {jazyk% : []}
        summary = {"dates":[], "avg_activity": {}}
        for date, profiles in sorted(self.evolution_stats["avg_activity"].items()):
            summary["dates"].append(date)
            
            for profile, count in profiles.items():
                if not PROFILES:
                    if profile not in summary["avg_activity"]:
                        summary["avg_activity"][profile] = []
                    summary["avg_activity"][profile].append(count)
                else:
                    if profile not in PROFILES:
                        continue
                    if profile not in summary["avg_activity"]:
                        summary["avg_activity"][profile] = []
                    summary["avg_activity"][profile].append(count)
                
        max_len = max(len(lst) for lst in summary["avg_activity"].values())
    
        summary["avg_activity"] = {
            key: [0] * (max_len - len(lst)) + lst
            for key, lst in summary["avg_activity"].items()
        }
                
            
        
        return f'''
            <div class="row">
                <div class="chart">{self.create_line_graph(
                    list(summary['dates']), 
                    [j for i,j in summary['avg_activity'].items()],
                    list(summary['avg_activity'].keys()),
                    f"Average activity evolution for {', '.join(PROFILES) if PROFILES else 'all profiles'}"
                )}</div>
            </div>  
        '''  

    def avg_activity_sum(self, PROFILES=[]):
        summary = {}
        for date, profiles in self.evolution_stats["avg_activity"].items():
            for profile, count in profiles.items():
                if profile not in summary:
                    summary[profile] = 0
                if not PROFILES:
                    summary[profile] += count
                elif profile in PROFILES:
                    summary[profile] += count

        return f'''
            <div class="row">
                <div class="chart">
                    <p>Average tweet rate of bubble: {sum(summary.values())} per month   </p>
                    <p>Average tweet rate per profile: {sum(summary.values())/len(set(PROFILES)&set(summary.keys()) if PROFILES else set(summary.keys()))} per month</p>
                </div>
            </div>  
        '''

    def locations_sum(self, PROFILES=[]):
        return f'''
            <div class="row">
                <div class="chart">{self.create_pie_chart(
                    list(self.overall_stats["locations"].keys()), 
                    [len(set(j) & set(PROFILES)) for i,j in self.overall_stats["locations"].items()], 
                    f"Locations of {', '.join(PROFILES) if PROFILES else 'all profiles'}"
                )}</div>
            </div>  
        '''
    def most_followed_profiles(self, PROFILES=[], threshold=0, show_number=float('inf')):
        summary = {}
        for i, j in self.social_bubble.get_outside_profiles_data(threshold).items():
            if PROFILES:
                summary[i] = len(set(PROFILES) & set(self.social_bubble.followed_outside_bubble[i]))
            else:
                summary[i] = len(self.social_bubble.followed_outside_bubble[i])
                
        summary = sorted(summary.items(), key=lambda x: x[1], reverse=True)
        summary = ', '.join([x[0] for x in summary][:min(len(summary), show_number)])
                
        return f'''
            <div class="row">
                <p>Most followed profiles outside bubble: {summary}</p>
            </div>  
        '''

    def item_usage(self, ITEM, PROFILES=[], show_number=float('inf')): # item is "hashtags" or "mentions"
        summary = {}
        for date, items in self.evolution_stats[ITEM].items():
            for item, data in items.items():
                if item not in summary:
                    summary[item] = 0
                for profile, count in data.items():
                    if not PROFILES:
                        summary[item] += count
                    elif profile in PROFILES:
                        summary[item] += count

        summary = dict(sorted(summary.items(), key=lambda x: x[1], reverse=True)[:min(show_number, len(summary))])

        return f'''
            <div class="row">
                <div class="chart">
                    {self.create_pie_chart(
                        list(summary.keys()), 
                        list(summary.values()), 
                        f"{' '.join(ITEM.capitalize().split('_'))} usage of {', '.join(PROFILES) if PROFILES else 'all profiles'}"
                    ) if 'sentiment' not in ITEM else self.create_bar_chart(
                        list(summary.keys()), 
                        [10*i for i in summary.values()], 
                        f"{' '.join(ITEM.capitalize().split('_'))} in {', '.join(PROFILES) if PROFILES else 'all profiles'}"
                    )}
                </div>
            </div>  
        '''

    def item_spread(self, ITEM, PROFILES=[], show_number=float('inf')):
        summary = {}
        for date, items in self.evolution_stats[ITEM].items():
            for item, data in items.items():
                if item not in summary:
                    summary[item] = set()
                if not PROFILES:
                    summary[item] |= {k for k, v in data.items() if v > 0}
                else:
                    summary[item] |= {k for k, v in data.items() if v > 0}&set(PROFILES)
        for key in summary.keys():
            summary[key] = len(summary[key])

        summary = dict(sorted(summary.items(), key=lambda x: x[1], reverse=True)[:min(show_number, len(summary))])

        return f'''
            <div class="row">
                <div class="chart">
                    {self.create_pie_chart(
                        list(summary.keys()), 
                        list(summary.values()), 
                        f"{' '.join(ITEM.capitalize().split('_'))} spread of {', '.join(PROFILES) if PROFILES else 'all profiles'}"
                    )}
                </div>
            </div>  
        '''

    def item_evolution(self, ITEM, PROFILES=None):
        #print(self.evolution_stats[ITEM])
        # {jazyk% : []}
        summary = {"dates":[], ITEM: {}}
        for date, item in sorted(self.evolution_stats[ITEM].items()):
            summary["dates"].append(date)
            
            for item, profiles in item.items():
                if item not in summary[ITEM]:
                    summary[ITEM][item] = []
                if PROFILES:
                    summary[ITEM][item].append(sum([j for i,j in profiles.items() if i in PROFILES]))
                else:
                    summary[ITEM][item].append(sum(profiles.values()))
                
        max_len = max(len(lst) for lst in summary[ITEM].values())
    
        summary[ITEM] = {
            key: [0] * (max_len - len(lst)) + lst
            for key, lst in summary[ITEM].items()
        }
                
            
        
        return f'''
            <div class="row">
                <div class="chart">{self.create_line_graph(
                    list(summary['dates']), 
                    [j for i,j in summary[ITEM].items()],
                    list(summary[ITEM].keys()),
                    f"{' '.join(ITEM.capitalize().split('_'))} evolution for {', '.join(PROFILES) if PROFILES else 'all profiles'}"
                )}</div>
            </div>  
        '''

    def other_topics(self, PROFILES=[], show_number=float('inf')):
        summary = {}
##        print(self.overall_stats["other_topics"])
        merged = AI_GENERALISATION_PARSER.generalise(self.overall_stats["other_topics"])
##        print(merged)
        for i in merged["list_of_topics"]:
            if i["interest"] not in summary:
                summary[i["interest"]] = 0
            for j in i["map_counter"]:
                if PROFILES and j["profile_name"] not in PROFILES:
                    continue
                summary[i["interest"]] += j["counter"]

        summary = dict(sorted(summary.items(), key=lambda x: x[1], reverse=True)[:min(show_number, len(summary))])
                
        return f'''
            <div class="row">
                <div class="chart">
                    {self.create_pie_chart(
                        list(summary.keys()), 
                        list(summary.values()), 
                        f"Topics of interest for {', '.join(PROFILES) if PROFILES else 'all profiles'}"
                    )}
                </div>
            </div>  
        '''
    def other_topics_spread(self, PROFILES=[], show_number=float('inf')):
        summary = {}
##        print(self.overall_stats["other_topics"])
        merged = AI_GENERALISATION_PARSER.generalise(self.overall_stats["other_topics"])
##        print(merged)
        for i in merged["list_of_topics"]:
            if i["interest"] not in summary:
                summary[i["interest"]] = 0
            if PROFILES:
                summary[i["interest"]] += len(set([j['profile_name'] for j in i["map_counter"]])&set(PROFILES))
            else:
                summary[i["interest"]] += len([j['profile_name'] for j in i["map_counter"]])

        summary = dict(sorted(summary.items(), key=lambda x: x[1], reverse=True)[:min(show_number, len(summary))])
##        print(summary)
        return f'''
            <div class="row">
                <div class="chart">
                    {self.create_pie_chart(
                        list(summary.keys()), 
                        list(summary.values()), 
                        f"Topics of interest for {', '.join(PROFILES) if PROFILES else 'all profiles'}"
                    )}
                </div>
            </div>  
        ''' 

    def compass(self, PROFILES=[]):
        summary = []
        for name, (x,y,r) in self.overall_stats["compass"].items():
            if PROFILES and name not in PROFILES:
                continue
            if r:
                summary.append((x/r*10, y/r*10, name, 'black'))
            
        avg_x = sum([x[0] for x in summary])/len(summary)
        avg_y = sum([y[1] for y in summary])/len(summary)
        summary.append((avg_x, avg_y, 'Average', 'yellow'))
        
        return f'''
            <div class="row">
                <div class="chart">
                    {self.create_grid_chart(summary)}
                </div>
            </div>  
        '''

    def ideology_usage(self, PROFILES=[], show_number=float('inf')):
        summary = {}
        for item, data in self.overall_stats['ideologies_overall'].items():
            if item not in summary:
                summary[item] = 0
            for profile, count in data.items():
                if not PROFILES:
                    summary[item] += count
                elif profile in PROFILES:
                    summary[item] += count

        summary = dict(sorted(summary.items(), key=lambda x: x[1], reverse=True)[:min(show_number, len(summary))])

        return f'''
            <div class="row">
                <div class="chart">
                    {self.create_pie_chart(
                        list(summary.keys()), 
                        list(summary.values()), 
                        f"Prefered ideologies of {', '.join(PROFILES) if PROFILES else 'all profiles'}"
                    )}
                </div>
            </div>  
        '''
    
        
    

        

    def test_show(self):
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
                    height: 700px;
                    overflow-x: auto; /* Allows horizontal scrolling if content overflows */
                    white-space: nowrap; /* Prevents items from wrapping to next line */
                    padding: 10px 0; /* Optional: adds some vertical padding */
                }}
                .chart {{
                    flex: 0 0 auto; /* Prevents charts from shrinking/growing */
                    width: 550px; /* Fixed width for each chart */
                    height: 550px; /* Fixed height for each chart */
                }}
            </style>
        </head>
        <body>
            <div class="container">
                {self.compass()}

                {self.ideology_usage()}
            
                {self.other_topics([], 20)}

                {self.other_topics_spread([], 20)}
                
                {self.avg_activity_sum()}{self.avg_activity_sum(['pushkicknadusu'])}{self.avg_activity_evolution()}
                {self.locations_sum()}{self.locations_sum(['pushkicknadusu'])}
                {self.most_followed_profiles([], 100, 5)}
                
                {[self.item_spread(i, [], 5) for i in list(set(self.evolution_stats.keys())-{'avg_activity'}) if self.evolution_stats[i] and 'overall' not in i]}
                {[self.item_usage(i, [], 5) for i in list(set(self.evolution_stats.keys())-{'avg_activity'}) if self.evolution_stats[i] and 'overall' not in i]}
                {[self.item_evolution(i) for i in list(set(self.evolution_stats.keys())-{'avg_activity'}) if self.evolution_stats[i] and 'overall' not in i]}
                
            </div>
        </body>
        </html>
        '''

        
        with open("test.html", "w", encoding="utf-8") as file:
            file.write(html_content)
        print(f"HTML file has been created.")

    def graph_properties(self, PROFILES=[]):
        traffic = {}
        interconnection = {}
        top_conversation = [None, 0]
        one_dir_edges = 0
        both_dir_edges = 0
        
        
        for name, node in self.social_bubble.nodes.items():
            traffic[name] = node.compute_interaction_strength(PROFILES)
            interconnection[name] = len(node.edges)

        for edge in self.social_bubble.edges:
            weight = edge.get_weight_eval()
            if top_conversation[1] >= weight:
                if PROFILES and (edge.node1 not in PROFILES or edge.node2 not in PROFILES):
                    continue
                top_conversation = [edge, weight]
                
            if edge.weight["follows"] in ["->", "<-"]:
                one_dir_edges += 1
            if edge.weight["follows"] == 'friends':
                both_dir_edges += 1
                
            
        traffic = dict(sorted(traffic.items(), key=lambda x: x[1], reverse=True))
        interconnection = dict(sorted(interconnection.items(), key=lambda x: x[1], reverse=True))

        density_one_dir = (one_dir_edges+2*both_dir_edges)/(math.factorial(len(self.social_bubble.nodes))/math.factorial(len(self.social_bubble.nodes)-2))
        density_both_dir = both_dir_edges/(math.factorial(len(self.social_bubble.nodes))/(math.factorial(len(self.social_bubble.nodes)-2)*2))
        
        self.graph_prop = {'traffic':traffic, 'interconnection':interconnection, 'top_conversation':top_conversation, 'density_one_dir':density_one_dir, 'density_both_dir':density_both_dir}
        return self.graph_prop

    def create_entity_based_graph(self, topic, entity, intervals=False):
        
        if not intervals:
            net = Network(notebook=True, height="600px", width="100%", bgcolor="#222222", font_color="white")
            active_nodes = {i:{'sentiment': 0, 'mentions':0} for i in self.social_bubble.nodes.keys()}
            data = self.evolution_stats.get(f"{topic}_overall_sentiment", False)
            if data:
                if entity in data:
                    for username, sentiment in data[entity].items():
                        active_nodes[username] = {'sentiment': sentiment, 'mentions':0}
                else:
                    print(f"NO SENTIMENT DATA FOUND FOR ENTITY {entity}")
            
            data = self.evolution_stats.get(f"{topic}_overall_mentions", False)
            if data:
                if entity in data:
                    for username, mention in data[entity].items():
                        if username not in active_nodes:
                            active_nodes[username] = {'sentiment': 0, 'mentions':0}
                        active_nodes[username]['mentions'] = mention
                else:
                    print(f"NO MENTIONS DATA FOUND FOR ENTITY {entity}")
            else:
                if entity in self.overall_stats["other_topics"]:
                    for username, mention in data[entity].items():
                        if username not in active_nodes:
                            active_nodes[username] = {'sentiment': 0, 'mentions':0}
                        active_nodes[username]['mentions'] = mention
                else:
                    print(f"NO MENTIONS DATA FOUND FOR ENTITY {entity}")

            for username, data in active_nodes.items():
                sentiment = data['sentiment']
                mentions = data['mentions']
                
                # Color logic: Red (< -0.2), Yellow (-0.2 to 0.2), Green (> 0.2)
                if not mentions:
                    color = "#c0c0c0"
                elif sentiment < -0.2:
                    color = "#ff6666"  # Red
                elif -0.2 <= sentiment <= 0.2:
                    color = "#ffcc00"  # Yellow
                else:
                    color = "#66ff66"  # Green
                
                # Size logic: Base size = 10, scales with mentions (min size = 10)
                size = 10 + (1.3 * mentions) if mentions > 0 else 10
                
                net.add_node(
                    username,
                    title=f"Sentiment: {sentiment:.2f}\nMentions: {mentions}",
                    color=color,
                    size=size
                )
            
            # --- 4. Add edges from social_bubble with STRICT sentiment coloring ---
            for edge in self.social_bubble.edges:
                node1 = edge.node1.profile.username
                node2 = edge.node2.profile.username
                if node1 in active_nodes and node2 in active_nodes:
                    # Count reaction sentiments
                    reaction_sentiments = {'Disagreeing': 0, 'Agreeing': 0, 'Neutral': 0}
                    
                    for tweet in edge.weight['reactions']["1->2"] + edge.weight['reactions']["2->1"]:
                        if tweet.content:
                            # Case 1: Direct reaction matches topic
                            if tweet.content.get('reaction', {}).get('type') == topic:
                                sentiment = tweet.content.get("reaction_sentiment")
                                if sentiment in reaction_sentiments:
                                    reaction_sentiments[sentiment] += 1
                            
                            # Case 2: Source tweet matches topic
                            elif tweet.source_tweet and ALL_TWEETS.get(tweet.source_tweet, None):
                                if ALL_TWEETS.get(tweet.source_tweet, None).content:
                                    source_type = ALL_TWEETS[tweet.source_tweet].content.get('type')
                                    source_reaction_type = ALL_TWEETS[tweet.source_tweet].content.get('reaction', {}).get('type')
                                    if source_type == topic or source_reaction_type == topic:
                                        sentiment = tweet.content.get("reaction_sentiment")
                                        if sentiment in reaction_sentiments:
                                            reaction_sentiments[sentiment] += 1

                    # STRICT dominant sentiment logic (requires 2x majority)
                    total = sum(reaction_sentiments.values())
                    edge_color = "#c0c0c0"
                    
                    if total > 0:
                        agreeing_ratio = reaction_sentiments['Agreeing'] / total
                        disagreeing_ratio = reaction_sentiments['Disagreeing'] / total
                        
                        # Only color if one sentiment has 2x more than the other
                        if agreeing_ratio >= 2 * disagreeing_ratio:
                            edge_color = "#66ff66"  # Green (Clear Agreeing majority)
                        elif disagreeing_ratio >= 2 * agreeing_ratio:
                            edge_color = "#ff6666"  # Red (Clear Disagreeing majority)
                        else:
                            edge_color = "#ffcc00"
                    
                    # Add edge with determined color
                    net.add_edge(
                        node1, 
                        node2, 
                        width=0.5,
                        color=edge_color,
                        title=f"Reactions:\n"
                             f"Agreeing: {reaction_sentiments['Agreeing']}\n"
                             f"Neutral: {reaction_sentiments['Neutral']}\n"
                             f"Disagreeing: {reaction_sentiments['Disagreeing']}"
                    )
            
            
            
            return net 
            
        else:
            graphs = {}
            for date, data in self.evolution_stats.get(f"{topic}_tweet_sentiment", {}).items():
                net = Network(notebook=True, height="600px", width="100%", bgcolor="#222222", font_color="white")
                active_nodes = {i:{'sentiment': 0, 'mentions':0} for i in self.social_bubble.nodes.keys()}
                if data:
                    if entity in data:
                        for username, sentiment in data[entity].items():
                            active_nodes[username] = {'sentiment': sentiment, 'mentions':0}
                    else:
                        print(f"NO SENTIMENT DATA FOUND FOR ENTITY {entity} DURING {date}")
                
                data_mentions = self.evolution_stats.get(f"{topic}_tweet_mentions", {}).get(date, False)
                if data_mentions:
                    if entity in data_mentions:
                        for username, mention in data_mentions[entity].items():
                            if username not in active_nodes:
                                active_nodes[username] = {'sentiment': 0, 'mentions':0}
                            active_nodes[username]['mentions'] = mention
                    else:
                        print(f"NO MENTIONS DATA FOUND FOR ENTITY {entity} DURING {date}")
                
                for username, data in active_nodes.items():
                    sentiment = data['sentiment']
                    mentions = data['mentions']
                    
                    # Color logic: Red (< -0.2), Yellow (-0.2 to 0.2), Green (> 0.2)
                    if not mentions:
                        color = "#c0c0c0"
                    elif sentiment < -0.2:
                        color = "#ff6666"  # Red
                    elif -0.2 <= sentiment <= 0.2:
                        color = "#ffcc00"  # Yellow
                    else:
                        color = "#66ff66"  # Green
                    
                    # Size logic: Base size = 10, scales with mentions (min size = 10)
                    size = 10 + (1.3 * mentions) if mentions > 0 else 10
                    
                    net.add_node(
                        username,
                        title=f"Sentiment: {sentiment:.2f}\nMentions: {mentions}",
                        color=color,
                        size=size
                    )
                
                # --- 4. Add edges from social_bubble with STRICT sentiment coloring ---
                for edge in self.social_bubble.edges:
                    node1 = edge.node1.profile.username
                    node2 = edge.node2.profile.username
                    if node1 in active_nodes and node2 in active_nodes:
                        # Count reaction sentiments
                        reaction_sentiments = {'Disagreeing': 0, 'Agreeing': 0, 'Neutral': 0}
                        
                        for tweet in edge.weight['reactions']["1->2"] + edge.weight['reactions']["2->1"]:
                            if tweet.content:
                                # Case 1: Direct reaction matches topic
                                if tweet.content.get('reaction', {}).get('type') == topic:
                                    sentiment = tweet.content.get("reaction_sentiment")
                                    if sentiment in reaction_sentiments:
                                        reaction_sentiments[sentiment] += 1
                                
                                # Case 2: Source tweet matches topic
                                elif tweet.source_tweet and ALL_TWEETS.get(tweet.source_tweet, None):
                                    if ALL_TWEETS.get(tweet.source_tweet, None).content:
                                        source_type = ALL_TWEETS[tweet.source_tweet].content.get('type')
                                        source_reaction_type = ALL_TWEETS[tweet.source_tweet].content.get('reaction', {}).get('type')
                                        if source_type == topic or source_reaction_type == topic:
                                            sentiment = tweet.content.get("reaction_sentiment")
                                            if sentiment in reaction_sentiments:
                                                reaction_sentiments[sentiment] += 1

                        # STRICT dominant sentiment logic (requires 2x majority)
                        total = sum(reaction_sentiments.values())
                        edge_color = "#c0c0c0"
                        
                        if total > 0:
                            agreeing_ratio = reaction_sentiments['Agreeing'] / total
                            disagreeing_ratio = reaction_sentiments['Disagreeing'] / total
                            
                            # Only color if one sentiment has 2x more than the other
                            if agreeing_ratio >= 2 * disagreeing_ratio:
                                edge_color = "#66ff66"  # Green (Clear Agreeing majority)
                            elif disagreeing_ratio >= 2 * agreeing_ratio:
                                edge_color = "#ff6666"  # Red (Clear Disagreeing majority)
                            else:
                                edge_color = "#ffcc00"
                        
                        # Add edge with determined color
                        net.add_edge(
                            node1, 
                            node2, 
                            width=0.5,
                            color=edge_color,
                            title=f"Reactions:\n"
                                 f"Agreeing: {reaction_sentiments['Agreeing']}\n"
                                 f"Neutral: {reaction_sentiments['Neutral']}\n"
                                 f"Disagreeing: {reaction_sentiments['Disagreeing']}"
                        )
                
                # --- 5. Finalize and save graph ---
                graphs[date] = net
                
            return graphs
                #net.show(f"{entity}_network.html")
                
                
            


        for topic in ["sport", "club", "artist", "genre", "athlete"]:
            print(topic)
            print(self.evolution_stats[f"{topic}_tweet_sentiment"])
            print(self.evolution_stats[f"{topic}_overall_sentiment"])
            print(self.evolution_stats[f"{topic}_tweet_mentions"])
            print(self.evolution_stats[f"{topic}_overall_mentions"])

    ##### entity moze byt none iba pri mentions
    
    def interactions_subbubbles(self):
        # Prepare graph data
        nodes = list(self.social_bubble.nodes.keys())
        edges = {}
        for edge in self.social_bubble.edges:
            edges[(edge.node1.profile.username, edge.node2.profile.username)] = edge.get_weight_eval()

        # Create networkx graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from((u, v, {'weight': w}) for (u, v), w in edges.items())

        # Detect communities (including single-node communities)
        partition = community_louvain.best_partition(G, weight='weight', randomize=False)
        
        # Ensure all nodes are in partition (some might be isolated)
        for node in nodes:
            if node not in partition:
                partition[node] = max(partition.values()) + 1 if partition else 0

        # Generate visually distinct colors
        community_colors = {}
        for comm_id in set(partition.values()):
            hue = comm_id / max(1, len(set(partition.values())))  # Evenly distribute hues
            community_colors[comm_id] = f"hsl({int(hue*360)}, 80%, 60%)"

        # Create PyVis network
        net = Network(notebook=True, height="800px", width="100%", 
                     bgcolor="#222222", font_color="white", directed=False)
        
        # Add nodes with community colors
        for node in nodes:
            community_id = partition.get(node, -1)
            net.add_node(
                node,
                color=community_colors.get(community_id, "#888888"),
                title=f"User: {node}<br>Community: {community_id}",
                size=15  # Base size for all nodes
            )
        
        # Add edges with special treatment for inter-community connections
        for (node1, node2), weight in edges.items():
            comm1 = partition.get(node1, -1)
            comm2 = partition.get(node2, -1)
            
            # Determine edge properties
            if comm1 == comm2:
                # Intra-community edge
                edge_color = community_colors.get(comm1, "#888888")
                edge_length = 100  # Shorter length for same-group connections
            else:
                # Inter-community edge
                edge_color = "#aaaaaa"
                edge_length = 200 + (1-weight)*100  # Longer for weaker connections
            
            net.add_edge(
                node1,
                node2,
                color=edge_color,
                value=weight,
                title=f"Interaction: {weight:.3f}",
                length=edge_length,
                width=0.5 + weight * 3,
                smooth={'enabled': True, 'type': 'continuous'}
            )
        
        # Configure physics for better group separation
        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -5000,
              "centralGravity": 0.1,
              "springLength": 150,
              "springConstant": 0.01,
              "damping": 0.2
            },
            "minVelocity": 0.75
          }
        }
        """)
        
        # Add community legend
        legend_html = """
        <div style="position: absolute; top: 10px; left: 10px; z-index: 1000; 
                    background: rgba(40,40,40,0.8); color: white; padding: 10px; border-radius: 5px;">
        <b>Community Legend</b><br>
        """
        for comm_id, color in community_colors.items():
            legend_html += f'<span style="color:{color}">■</span> Community {comm_id}<br>'
        legend_html += '<span style="color:#aaaaaa">■</span> Between communities</div>'
        
        net.html = legend_html + net.html
        
        # Save and return
        net.show("communities.html")
        return net
            
         
'''
DECENTRLISED
CONSOLE WINDOW
PODBUBLINY
'''    
        



SB = SocialBubble("pushkicknadusu", "decentralised", depth=3, profiles=["pushkicknadusu", "statkar_miky"])

SB.create_graph()

##################################################BS = BubbleSummary({}, SB)
##################################################
####################################################BS.interactions_subbubbles()

SB.visualize_graph()

SB.visualize_outside_relations()

##SB.visualize_hashtags()

##opd = SB.get_outside_profiles_data(THRESHOLD)

##print(len(opd))

##SB.profile_analysis(opd)

####SB.tweet_analysis()

####BS = BubbleSummary(SB.profiles_summary(6), SB)

####BS.test_show()

##print(BS.graph_properties())




'''
1. VYTVORIT KVALITNU ANALYZU PROFILU + VYOBRAZENIE


mozne funkcie:
    UNION, INTERSECTION
    AK JE NEJAKA TOPIC, ZVYRAZNIT TIE KTORYCH SA TO TYKA + HRANY (INTERAKCIA, ROVNAKY FOLLOW NA DANU TOPIC ...)

    CONTAINS => prejdenie tweetov s tým, že sa analyzuje, či obsahuje danú tému a sentiment na ňu

    CROP -> oreze veci z bubliny podla volby

2. VYTVORIT SUHRN BUBLINY + ANALYZY
3. KONZOLOVA APLIKACIA + VIZUALIZACIE


'''
