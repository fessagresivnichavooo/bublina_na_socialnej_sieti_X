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
PROFILE_AI_ANALYSER = AIAnalysis.GPT4o_mini()
TWEET_AI_ANALYSER = None
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
                                    e.weight["mentions"][e.direction(n, self.nodes[i])] += n.all_mentions[i]

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


                                for i in list(set(n.profile.followed) & set(self.nodes.keys())):
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
                self.nodes[follower] = n
                self.edges.append(Edge("friends", base, n))
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
            
        





SB = SocialBubble("pushkicknadusu", "profile_centered", depth=2)

SB.create_graph()

SB.visualize_graph()

with open("profile_analysis.json", 'r', encoding="utf-8") as file:
    profiles = json.load(file)
    

for_analysis = SB.get_outside_profiles_data(1000)
for profile in profiles:
    if profile in for_analysis:
        for_analysis.pop(profile)
for_analysis = [(key, *value) for key, value in for_analysis.items()]

for profile_data in PROFILE_AI_ANALYSER.analyze_profiles(for_analysis):
    profiles[profile_data["screen_name"]] = profile_data

with open("profile_analysis.json", "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=4, ensure_ascii=False)


#print(SB.followed_outside_bubble)

#print(ALL_TWEETS)























data = [
    {
        "id": "1854867371346841934",
        "text": "Kurva mohli by na wolte pridať funkciu že miesto tringeltu kuriérovi strhneš z platu, by mi už aspoň nechodilo jedlo studené jak grónska prostitútka",
        "likes": 2,
        "replies": 1,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-11-08T12:43:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1854867371346841934"
    },
    {
        "id": "1851043656448332003",
        "text": "Legenda hovorí že chlapci čo dávajú pod fotky z chlastacky do popisu \"ACAB\" majú o 50% väčšiu šancu že po inkasovaní troch faciek podávajú trestné oznámenie",
        "likes": 1,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-10-28T23:29:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1851043656448332003"
    },
    {
        "id": "1849935670594384264",
        "text": "Už týždeň som nefukal parno (fajčil som ho)",
        "likes": 1,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-10-25T22:06:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1849935670594384264"
    },
    {
        "id": "1849893993615851698",
        "text": "Feminizmus končí tam kde začína môj pravý direct",
        "likes": 1,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-10-25T19:21:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1849893993615851698"
    },
    {
        "id": "1847209432658149475",
        "text": "To keď sa snažím prestať tlačiť a niekto vytiahne puk začnem byť nervózny jak ITčkar keď je 10 metrov od skutočnej ženy",
        "likes": 1,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-10-18T09:33:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1847209432658149475"
    },
    {
        "id": "1845481075713118249",
        "text": "Môžeš byť najviac prepnutý sociopat a furt sa ťa budem báť menej ako čáva čo dáva vyžúvané sakle naspäť do puku",
        "likes": 1,
        "replies": 0,
        "retweets": 1,
        "quotes": 0,
        "timestamp": "2024-10-13T15:05:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1845481075713118249"
    },
    {
        "id": "1845125945729949903",
        "text": "Čajky sú jak doberman, to ak nie si na to tvrdý raz za čas tak za pár rokov ťa dojebe lebo nemá k tebe rešpekt",
        "likes": 2,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-10-12T15:34:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1845125945729949903"
    },
    {
        "id": "1843933306762735700",
        "text": "Ľudia čo vypisujú na nejakého chalanka že RIP budeš chýbať,kkt bavili ste sa 3 krát v živote,kebyže ti to je ľúto tak to nevypisuješ po ig. Ešte sa cítia street že poznajú po mene chlapca čo zomrel,ronia nad ním slzy pred publikom hlavne že rodine ani sviečku nezapália na dušičky",
        "likes": 3,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-10-09T08:35:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1843933306762735700"
    },
    {
        "id": "1841044146342993979",
        "text": "Vysvetlite mi niekto prečo ľudia chodia k terapeutom. Čavovi platíš 30€ na hodinu že sa s tebou rozpráva o tom že ti je napiču, to fakt že nikto si nemôže želať viac aby si bol nešťastný jak on. Kebyže mám 30€/h za deratizáciu pivnice tak tam nechám potkany behať ešte 10 rokov",
        "likes": 1,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-10-01T09:15:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1841044146342993979"
    },
    {
        "id": "1833680038304129357",
        "text": "Kyrgystanka nechce výjsť dverami, tratí na úkor Changov",
        "likes": 1,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-09-11T01:32:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1833680038304129357"
    },
    {
        "id": "1833593654549913920",
        "text": "Tratí sa na úkor Changov",
        "likes": 2,
        "replies": 1,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-09-10T19:49:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1833593654549913920"
    },
    {
        "id": "1832531593568927870",
        "text": "Konečne kurva čalamáda už líže textil ale tak aspoň za kurvami nezdrháš",
        "likes": 1,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-09-07T21:29:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1832531593568927870"
    },
    {
        "id": "1820596991878889958",
        "text": "Dnes sa spýtaš týpka odkiaľ je a každý druhý ti zajebe \"821\" alebo také niečo lebo sa tvári street.\nKkt veď povedz Ružinov čo som ja nejaký poštár dpc",
        "likes": 4,
        "replies": 0,
        "retweets": 1,
        "quotes": 0,
        "timestamp": "2024-08-05T23:05:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1820596991878889958"
    },
    {
        "id": "1806219219979076014",
        "text": "Ctjb tento rok Slovákom vydrží patriotizmus možno že až do štvrťfinále",
        "likes": 3,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-06-27T06:53:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1806219219979076014"
    },
    {
        "id": "1795769311312962001",
        "text": "Fuj ty kkt nejaký liberál si vedľa mňa sadol v električke, asi si presadnem k bezdomovcovi",
        "likes": 4,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-05-29T10:49:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1795769311312962001"
    },
    {
        "id": "1793656591004356766",
        "text": "Kkti už mi nepúšťajte toľko hokeja do tých reklám",
        "likes": 2,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-05-23T14:53:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1793656591004356766"
    },
    {
        "id": "1792803027658944797",
        "text": "Auto je zbraň a zbrane ženám do rúk nepatria",
        "likes": 2,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-05-21T06:22:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1792803027658944797"
    },
    {
        "id": "1783793223535603805",
        "text": "Vyzývam ťa východniar! Ešte raz to skús! Skús mi v piatok na chodbe ešte raz vojsť do cesty s tým tvojim pojebaným kufrom !",
        "likes": 2,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-04-26T09:40:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1783793223535603805"
    },
    {
        "id": "1778723039900828132",
        "text": "Drahí ITčkari, za to že niekto došiel v kraťasoch a krátkom tričku keď je vonku 5 stupňov a tváril sa že mu není zima ešte nikdy nikomu nijaká neskočila na kkt. Ja len tak",
        "likes": 1,
        "replies": 1,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-04-12T09:53:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1778723039900828132"
    },
    {
        "id": "1768635325188915680",
        "text": "Jebnem si acid pod viečko nech pozorujem vašu obmedzenosť s nadhľadom",
        "likes": 2,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-03-15T13:48:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1768635325188915680"
    },
    {
        "id": "1763949979498877010",
        "text": "Pri zakladaní Instagram účtu by mal byť povinný IQ test",
        "likes": 3,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-03-02T15:30:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1763949979498877010"
    },
    {
        "id": "1760084757180731392",
        "text": "Kto kurva nechá 200€ v bordeli keď vstup do trafa stojí 10€",
        "likes": 3,
        "replies": 1,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-02-20T23:31:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1760084757180731392"
    },
    {
        "id": "1756396126574391518",
        "text": "Pred hipisáckymi podnikmi by mali byť povinné sprchy jak do plavárne",
        "likes": 3,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-02-10T19:14:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1756396126574391518"
    },
    {
        "id": "1750856144958926993",
        "text": "To fakt existujú 20+ roční chlapci čo sú presvedčení že papať liečiky je cool?",
        "likes": 2,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-01-26T12:20:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1750856144958926993"
    },
    {
        "id": "1747379584297742360",
        "text": "Frajer ti nekúpil kaser aby si ho nosila zahrabaný na spodku kabelky keď ideš von",
        "likes": 3,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-01-16T22:05:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1747379584297742360"
    },
    {
        "id": "1746627331643298206",
        "text": "Moc nízky tlak mám, idem si prečítať refresher",
        "likes": 3,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-01-14T20:16:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1746627331643298206"
    },
    {
        "id": "1741675493319929987",
        "text": "Aaaaa šťastný nový rok jakí ste zrazu poctiví všetci",
        "likes": 2,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2024-01-01T04:19:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1741675493319929987"
    },
    {
        "id": "1730521787656450183",
        "text": "Chlapci budú vyprávať o thinking outside the box predtým jak sa prefúkajú na dno",
        "likes": 2,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2023-12-01T09:38:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1730521787656450183"
    },
    {
        "id": "1730183475670442406",
        "text": "Fuuu ctjb od šťastia som roztečený jak na výlete, som že v top 0.0001% weeknd fans na spotify, asi si jebnem jedného huberta oslavného",
        "likes": 1,
        "replies": 1,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2023-11-30T11:14:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1730183475670442406"
    },
    {
        "id": "1729460845640327389",
        "text": "Dneska ľuďom nastavíš zrkadlo a oni na ňom začnú rysovať",
        "likes": 3,
        "replies": 0,
        "retweets": 0,
        "quotes": 0,
        "timestamp": "2023-11-28T11:22:00.000Z",
        "url": "https://x.com/pushkicknadusu/status/1729460845640327389"
    }
]

    
