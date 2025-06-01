class TwitterScrapper:
    def get_followers(self, username : str) -> list:
        ### FUNKCIA VRACIA LIST S MENAMI SLEDOVATEĽOV PROFILU USERNAME
        ### username = "elonmusk", output => ["tesla", "spaceX", ...]      

    def get_following(self, username : str) -> list:
        ### FUNKCIA VRACIA LIST S MENAMI SLEDOVANÝCH PROFILOV PROFILU USERNAME
        ### username = "elonmusk", output => ["tesla", "spaceX", ...]

    def get_following_extended(self, username : str) -> list:
        following = []
        for i in self.get_following(username):
            following.append([i, self.get_fullName(i), self.get_followers_count(i), self.get_bio(i)])
        return following        

    def scrape_profile(self, username):
        return (self.get_bio(username), self.get_followers(username), self.get_following(username), 
            self.get_location(username), self.get_fullName(username))

    def get_tweets(self, username : str) -> list:
        ### FUNKCIA VRACIA LIST SPRACOVANÝCH TWEETOV
        ### JEDEN TWEET => (ID, username, text, typ, source tweet, source username, hashtags, mentions, created)
        ### Typ može byť: "quote", "comment", "repost", "tweet"
        ### Ak je tweet reakcia na iný tweet, source tweet je jeho ID
        ### a source username jeho meno
        ### Hashtags a mentions => množiny
        ### Created => dátum vytvorenia
    
    def get_followers_count(self, username : str) -> int:
        ### VRACIA POČET FOLLOWEROV

    def get_fullName(self, username : str) -> str:
        ### VRACIA MENO UŽÍVATEĽA
    
    def get_location(self, username : str) -> str:
        ### VRACIA LOKÁCIU ZADANÚ V POPISE PROFILU

    def get_bio(self, username : str) -> str:
        ### VRACIA OBSAH BIA (POPISU PROFILU)
