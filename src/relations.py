import copy
import networkx as nx
import matplotlib.pyplot as plt
import test_twitter_scrapper_from_json
import AIAnalysis
import json

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

    def direction(self, node1: Node, node2: Node):
        if node1 == self.node1 and node2 == self.node2:
            return "1->2"
        elif node2 == self.node1 and node1 == self.node2:
            return "2->1"
        else:
            raise ValueError("invalid node parameter")

    def __str__(self):
        return f"{self.node1.profile.username}; {self.node2.profile.username} | {self.weight}"



        


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
                
    def analyse_tweets(self):
        for tweet in self.tweets:
            tweet.analyse()
        
    def __repr__(self):
        return (
            f"Profile(username={self.username!r}, profession={self.profession!r}, location={self.location!r})\n"
            f"Bio: {self.bio}\n"
            f"Followers: {len(self.followers)}, Following: {len(self.following)}, Friends: {len(self.friends)}\n"
            f"Tweets: {len(self.tweets)}, Reposts: {len(self.reposts)}, Comments: {len(self.comments)}, Quotes: {len(self.quotes)}\n"
##            f"Recent Tweets:\n" +
##            "\n".join(f"  - {tweet.text}" for tweet in self.tweets[:5])  # Show only first 5 tweets for readability
        )

    def set_profession(self):
        pass
        ### pomocou chatgpt ziska profesiu profilu
        ### sport => klub, krajina ; hudba => zanre ; politika ; ostatne


### repost retweet hashtagy 
class Tweet:
    def __init__(self, source_id, username, text, type, source_tweet=None, source_username=None, hashtags=[], mentions=[]):
        self.source_id = source_id
        self.username = username
        self.text = text
        self.type = type
        self.source_tweet = source_tweet
        self.source_username = source_username
        self.hashtags = hashtags
        self.mentions = mentions
        self.date = None
        self.content = None

    def analyse(self):
        if self.type == "tweet":
            self.content = TWEET_AI_ANALYSER.analyze_tweet(self.text)
            print(self.text, '\n', self.content, '\n')

    def __repr__(self):
        return f"""
            SOURCE ID: {self.source_id}
            USER: {self.username}
            CONTENT: {self.text}
            TYPE: {self.type}
            SOURCE TWEET: {ALL_TWEETS[self.source_tweet].text if self.source_tweet in ALL_TWEETS else None}
            SOURCE USER: {self.source_username}
            HASHTAGS: {self.hashtags}
            MENTIONS: {self.mentions}

        """
    def __str__(self):
        return self.text

    def get_type(self):
        return self.type

    def check_topic(self):
        return ""
    




        







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
                                self.interacted_outside_bubble.append(tweet.source_username)
                        

                
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
        for edge in self.edges:
            print(edge)

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
        #for node in self.nodes.values():
        #    node.profile.analyse_tweets()
        for tweet in ALL_TWEETS.values():
            tweet.analyse()


    def profile_analysis(self, profiles, cache="profile_analysis.json"):
        with open(cache, 'r', encoding="utf-8") as file:
            cached_profiles = json.load(file)

        print(profiles)

        serpapi = AIAnalysis.SerpAPI()

        for username, profile in profiles.items(): #(screen_name: name, description)
            if username in cached_profiles.keys():
                continue
            entity_data = serpapi.get_entity(profile[0])
            serpapi_formated = serpapi.process_entity(entity_data)
            serpapi_formated["Twitter username"] = username
            serpapi_formated["Twitter bio"] = profile[1]
            analysis = PROFILE_AI_ANALYSER.analyze_profile_II(serpapi_formated)
            cached_profiles[username] = analysis

        with open("profile_analysis.json", "w", encoding="utf-8") as f:
            json.dump(cached_profiles, f, indent=4, ensure_ascii=False)
        ###     TREBA OPRAVIT FORMAT CO IDE DO SUBORU





            
        

''' hrany s interakciami ohladom konkretnej temy '''



SB = SocialBubble("pushkicknadusu", "profile_centered", depth=0)

SB.create_graph()

SB.visualize_graph()

opd = SB.get_outside_profiles_data(2000)

print(opd)

SB.profile_analysis(opd)

#############   AK SERPAPI NEVRATI KVALITNY OUTPUT, PREFEROVAT BIO

#############   SKONTROLOVAT CI DESCRIPTION == BIO

#############   BUDE TREBA SKONTROLOVAT IMPLEMENTACIU CREATE_BUBBLE, CI
#############   NIEKDE NIE SU VYNECHANE UDAJE ATD





##SB.nodes.get("RobertFicoSVK", None).profile.analyse_tweets()



##with open("profile_analysis.json", 'r', encoding="utf-8") as file:
##    profiles = json.load(file)
##    
##
##for_analysis = SB.get_outside_profiles_data(1000)
##for profile in profiles:
##    if profile in for_analysis:
##        for_analysis.pop(profile)
##for_analysis = [(key, *value) for key, value in for_analysis.items()]
##
##
##with open("profile_analysis_SERPAPI.json", 'w', encoding="utf-8") as file:
##    
##
##for profile_data in PROFILE_AI_ANALYSER.analyze_profiles(for_analysis):
##    profiles[profile_data["screen_name"]] = profile_data
##
##with open("profile_analysis.json", "w", encoding="utf-8") as f:
##    json.dump(profiles, f, indent=4, ensure_ascii=False)


#print(SB.followed_outside_bubble)

#print(ALL_TWEETS)









##E = [0, 0, 0]
##S = [0, 0, 0]
##for i in SB.nodes.get("RobertFicoSVK", None).profile.tweets:
##    print(i.content)
##    try:
##        E[0] += i.content["politics"].get("E", 0)
##        E[1] += 1
##        if i.content["politics"].get("E", 0):
##            E[2] += 1
##        S[0] += i.content["politics"].get("S", 0)
##        S[1] += 1
##        if i.content["politics"].get("S", 0):
##            S[2] += 1
##    except AttributeError:
##        pass
##
##
##
##print(f"E: {E}, S: {S}")

#E: [-0.3000000000000001, 8], S: [1.9000000000000001, 8]
#E: [0.09999999999999998, 9], S: [2.3000000000000003, 9]

### vyskusat pomery dvoch osi

