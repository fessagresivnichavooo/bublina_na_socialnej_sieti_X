import json

data = {}
for i in range(2):
    with open(f"test_data_for_graph_{i+1}.json", "r", encoding="utf-8") as file:
        for profile, profile_data in json.load(file).items():
            data[profile] = profile_data


class TwitterScrapper:
    def __init__(self):
        pass
    #def get_bio(self, username):
        
    def get_followers(self, username):
        followers = []
        if username not in data:
            print(username)
            return []
        for i in data[username]["followers"]["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][3]["entries"]:
            try:
                followers.append(i["content"]["itemContent"]["user_results"]["result"]["legacy"]["screen_name"])
            except KeyError:
                continue
        if "verified_followers" in data[username]:
            for i in data[username]["verified_followers"]["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][3]["entries"]:
                try:
                    followers.append(i["content"]["itemContent"]["user_results"]["result"]["legacy"]["screen_name"])
                except KeyError:
                    continue
        return followers

    def get_following_extended(self, username):
        following = []
        if username not in data:
            return []
        for i in data[username]["following"]["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][3]["entries"]:
            try:
                screen_name = i["content"]["itemContent"]["user_results"]["result"]["legacy"]["screen_name"]
                name = i["content"]["itemContent"]["user_results"]["result"]["legacy"]["name"]
                followers = i["content"]["itemContent"]["user_results"]["result"]["legacy"]["followers_count"]
                description = i["content"]["itemContent"]["user_results"]["result"]["legacy"]["description"]
                
                following.append([screen_name, name, followers, description])
            except KeyError:
                continue
        return following
            

    def get_following(self, username):
        following = []
        if username not in data:
            return []
        for i in data[username]["following"]["data"]["user"]["result"]["timeline"]["timeline"]["instructions"][3]["entries"]:
            try:
                following.append(i["content"]["itemContent"]["user_results"]["result"]["legacy"]["screen_name"])
            except KeyError:
                continue
        return following

    def get_friends(self, username):
        return list(set(self.get_following(username)) & set(self.get_followers(username)))

##    def get_profession(self, username):
##
##    def get_location(self, username):
##
    def scrape_profile(self, username):
        return "NOT IMPLEMENTED", self.get_followers(username), self.get_following(username), "NOT IMPLEMENTED", self.get_fullName(username)

    def get_tweets(self, username):
        ## [text, type {repost, comment, tweet, quote}, ]
        tweets = []
        if username not in data:
            return []                
        for i in self.get_replies(username):
            if i not in tweets:
                tweets.append(i)
        
        return tweets
    
    def get_replies(self, username):
        replies = []
        if username not in data:
            return []
        a = data[username]["replies"]["data"]["user"]["result"].get("timeline_v2", None) or data[username]["replies"]["data"]["user"]["result"].get("timeline", None)        
        for entry in a["timeline"]["instructions"][-1]["entries"]:
            try:
                if "items" in entry["content"]:
                    # Ak odpoveď obsahuje zoznam položiek, prejdeme ich všetky
                    for item in entry["content"]["items"]:
                        tweet_data = item["item"]["itemContent"]["tweet_results"]["result"]["legacy"]
                        text = tweet_data["full_text"]
                        created = tweet_data["created_at"]
                        hashtags = [ht["text"] for ht in tweet_data["entities"].get("hashtags", [])]
                        mentions = [m["screen_name"] for m in tweet_data["entities"].get("user_mentions", [])]
                        source_tweet = None
                        source_username = None

                        if tweet_data.get("is_quote_status", False):
                            type = "quote"
                            temp = item["item"]["itemContent"]["tweet_results"]["result"].get("quoted_status_result", {}).get("result")
                            if temp:
                                source_username = temp["core"]["user_results"]["result"]["legacy"]["screen_name"]
                                source_tweet = item["item"]["itemContent"]["tweet_results"]["result"]["quoted_status_id_str"]
                        elif tweet_data.get("in_reply_to_screen_name", False):
                            type = "comment"
                            source_username = tweet_data["in_reply_to_screen_name"]
                            source_tweet = tweet_data["in_reply_to_status_id_str"]
                        elif "retweeted_status_result" in tweet_data:
                            type = "repost"
                            source_username = tweet_data["retweeted_status_result"]["result"]["core"]["user_results"]["result"]["legacy"]["screen_name"]
                            source_tweet = tweet_data["retweeted_status_result"]["result"]["rest_id"]
                        else:
                            type = "tweet"

                        replies.append((tweet_data["id_str"], item["item"]["itemContent"]["tweet_results"]["result"]["core"]["user_results"]["result"]["legacy"]["screen_name"], text, type, source_tweet, source_username, hashtags, mentions, created))

                else:
                    # Ak odpoveď nie je v zozname "items", spracujeme ju rovnako ako normálny tweet
                    tweet_data = entry["content"]["itemContent"]["tweet_results"]["result"]["legacy"]
                    text = tweet_data["full_text"]
                    created = tweet_data["created_at"]
                    hashtags = [ht["text"] for ht in tweet_data["entities"].get("hashtags", [])]
                    mentions = [m["screen_name"] for m in tweet_data["entities"].get("user_mentions", [])]
                    source_tweet = None
                    source_username = None

                    if tweet_data.get("is_quote_status", False):
                        type = "quote"
                        temp = entry["content"]["itemContent"]["tweet_results"]["result"].get("quoted_status_result", {}).get("result")
                        if temp:
                            source_username = temp["core"]["user_results"]["result"]["legacy"]["screen_name"]
                            source_tweet = item["item"]["itemContent"]["tweet_results"]["result"]["quoted_status_id_str"]
                    elif tweet_data.get("in_reply_to_screen_name", False):
                        type = "comment"
                        source_username = tweet_data["in_reply_to_screen_name"]
                        source_tweet = tweet_data["in_reply_to_status_id_str"]
                    elif "retweeted_status_result" in tweet_data:
                        type = "repost"
                        source_username = tweet_data["retweeted_status_result"]["result"]["core"]["user_results"]["result"]["legacy"]["screen_name"]
                        source_tweet = tweet_data["retweeted_status_result"]["result"]["rest_id"]
                    else:
                        type = "tweet"

                    replies.append((tweet_data["id_str"], entry["content"]["itemContent"]["tweet_results"]["result"]["core"]["user_results"]["result"]["legacy"]["screen_name"], text, type, source_tweet, source_username, hashtags, mentions, created))

            except Exception as e:
                print(f"Error processing reply: {e}")
                #print(entry)
                continue
        
        return replies


    def get_hashtags_from_tweet(self, tweet):
        return [i["text"] for i in tweet["content"]["itemContent"]["tweet_results"]["result"]["legacy"]["entities"]["hashtags"]]

    def get_mentions_from_tweet(self, tweet):
        return [i["screen_name"] for i in tweet["content"]["itemContent"]["tweet_results"]["result"]["legacy"]["entities"]["user_mentions"]]
    
    def get_followers_count(self, username):
        if username not in data:
            return None
        return data[username]["bio"]["data"]["user"]["result"]["legacy"]["followers_count"]

    def get_fullName(self, username):
        if username not in data:
            return None
        return data[username]["bio"]["data"]["user"]["result"]["legacy"]["name"]
        
