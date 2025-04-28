from pydantic import BaseModel, Field, RootModel
from typing import Dict, List, Optional, Union
import openai
import json
import requests


############# PYDANTIC #######################

# === Sport Models ===
class Athlete(BaseModel):
    type_of_sport: str = Field(..., description="Type of sport (e.g., Football, Basketball)")
    club: Optional[str] = Field(None, description="Club/team the athlete plays for")
    nationality: Optional[str] = Field(None, description="ISO 3166 code of nationality of the athlete")


class ClubTeam(BaseModel):
    sport_type: str = Field(..., description="Type of sport")
    country: Optional[str] = Field(None, description="ISO 3166 code of country where the club/team is based")

class SportFanpage(BaseModel):
    sport: str = Field(..., description="Sport related to the fanpage")


# === Music Models ===
class Author(BaseModel):
    genre: str = Field(..., description="Music genre of the author")
    country: Optional[str] = Field(None, description="ISO 3166 code of country where the author is influential")


class BandLabel(BaseModel):
    music_genre: str = Field(..., description="Music genre of the band/label")
    authors: Optional[List[str]] = Field(None, description="List of authors in the band/label")
    country: Optional[str] = Field(None, description="ISO 3166 code of country of influence")


class MusicFanpage(BaseModel):
    genre_of_music: str = Field(..., description="Music genre related to the fanpage")


# === Politics Models ===
class Politician(BaseModel):
    political_party: Optional[str] = Field(None, description="Political party the politician belongs to")
    ideology: Optional[str] = Field(None, description="Political ideology (liberalism, nationalism, conservatism, socialism, communism, environmentalism, social democracy, progressivism, anarchism, centrism, libertarianism, fascism, authoritarianism, religious-based ideology)")
    country: str = Field(..., description="ISO 3166 code of country where the politician is active")


class PoliticalPartyMovement(BaseModel):
    ideology: str = Field(..., description="Political ideology (liberalism, nationalism, conservatism, socialism, communism, environmentalism, social democracy, progressivism, anarchism, centrism, libertarianism, fascism, authoritarianism, religious-based ideology)")
    country: str = Field(..., description="ISO 3166 code of country where the political party/movement is active")    


class JournalistNews(BaseModel):
    country: str = Field(..., description="ISO 3166 code of country where the journalist/news is active")
    field: Optional[str] = Field(None, description="Political ideology if topic is politics, if sport return type of sport, if music return genre etc.")

# === Other Models ===

class Finance(BaseModel):
    sector: str = Field(..., description="Specific sector in finance (e.g., banking, investment, cryptocurrency)")
    country: Optional[str] = Field(None, description="ISO 3166 code of country where the financial organization or expert operates")


class Entertainment(BaseModel):
    industry: str = Field(..., description="Entertainment industry category (e.g., movies, gaming, television)")
    notable_work: Optional[str] = Field(None, description="Notable work, platform, or project related to entertainment")


class Technology(BaseModel):
    field: str = Field(..., description="Specific field in technology (e.g., AI, robotics, software, hardware)")
    notable_company: Optional[str] = Field(None, description="Major company or project associated with this entity")


class Education(BaseModel):
    specialization: str = Field(..., description="Field of education (e.g., STEM, humanities, business)")
    institution: Optional[str] = Field(None, description="Associated school, university, or institution")

class ArtAndCulture(BaseModel):
    field: str = Field(..., description="Field of art/culture (movie, painting, ...)")
    specification: str = Field(..., description="Genre/Type")
    
class Nature(BaseModel):
    specification: str = Field(..., description="Describe in 1 or 2 words")
    
class HobbiesFreeTime(BaseModel):
    field: str = Field(..., description="Type of hobby (travel, ...)")

class Other(BaseModel):
    topic: str = Field(..., description="Choose topic which defines the profile best")


# === Unified Topic Model ===
class TopicData(BaseModel):
    topic: str = Field(..., description="Main category, choose only from: Sport, Music, Politics, Other")
    type: str = Field(..., description="Specific entity type within the topic")
    data: Union[
        Athlete,
        ClubTeam,
        SportFanpage,
        Author,
        BandLabel,
        MusicFanpage,
        Politician,
        PoliticalPartyMovement,
        JournalistNews,
        Finance,
        Entertainment,
        Technology,
        Education,
        ArtAndCulture,
        Nature,
        HobbiesFreeTime,
        Other
    ]



class CountryDetails(BaseModel):
    country: str = Field(..., description="Associated country")
    clubs: Optional[List[str]] = Field(..., description="List of mentioned clubs of this country")
    athletes: Optional[List[str]] = Field(..., description="List of mentioned athletes of this country")


class SportEntry(BaseModel):
    sport: str = Field(..., description="Name of the sport, merge duplicates")
    counter: int = Field(..., description="How many times this sport is mentioned or appears")
    countries: List[CountryDetails] = Field(..., description="Mapping of country names to sport details")


class MusicEntry(BaseModel):
    genre: str = Field(..., description="Genre name")
    counter: int = Field(..., description="How many times this genre is mentioned or appears")
    countries: Optional[List[str]] = Field(..., description="List of countries mentioned in for this genre")
    artists: Optional[List[str]] = Field(..., description="Mentioned artists in this genre")

class Ideology(BaseModel):
    ideology: str = Field(..., description="Name of ideology")
    counter: int = Field(..., description="Number of mentions")

class Politics(BaseModel):
    ideologies: Optional[List[Ideology]] = Field(..., description="Mapping of ideology names to number of mentions (liberalism, nationalism, conservatism, socialism, communism, environmentalism, social democracy, progressivism, anarchism, centrism, libertarianism, fascism, authoritarianism, religious-based ideology)")
    countries: Optional[List[str]] = Field(..., description="ISO 3166 code of mentioned countries")

class OtherInterest(BaseModel):
    interest: str = Field(..., description="Name of topic/interest")
    counter: int = Field(..., description="Number of mentions")

class Interests(BaseModel):
    sport: Optional[List[SportEntry]] = Field(..., description="Mapping of sport name to sport-related data")
    music: Optional[List[MusicEntry]] = Field(..., description="Mapping of music genre to genre-related data")
    politics: Optional[Politics] = Field(..., description="Political ideologies and mentioned countries")
    other_interests: Optional[List[OtherInterest]] = Field(..., description='Other interests and how many times they were mentioned, choose most fitting from these: "Finance", "Entertainment", "Education", "Technology", "Science", "Health", "Art and Culture", "Hobbies", "Nature", "Other"')

class Counter(BaseModel):
    profile_name: str = Field(..., description="Rewrite profile name")
    counter: int = Field(...)

class Topic(BaseModel):
    interest: str = Field(..., description="Rewrite name of interest or return new generalised name after merging")
    map_counter: List[Counter] = Field(..., description="Maps profile name and count, when merging topics sum the counts for same profiles")
    
class OtherTopics(BaseModel):
    list_of_topics: List[Topic] = Field(..., description="Merge similar topics by finding super topic, which describes all found subtopics, mostly those with small count (ex. {Bitcoin : 1}, {Ethereum : 1} -> {Cryptocurrency : 2})")

###############################################################################






class GPT4o_mini():
    client = openai.OpenAI(api_key="

    class AnalysisProfileName(BaseModel):
        type: str = Field(..., description="Choose from following words to describe twitter profile: celebrity/fanpage/company/amusement-education/private profile")
        topic: str = Field(...,description="politics/music/sport/other")
        short_describtion: str = Field(...,description="Describe who it is")
        country: str = Field(...,description="Sphere of influence (example: if musician is born in russia but sing in france and in french, return 'france')")
        political_orientation: str = Field(..., description="If topic is politics, choose from: liberal/libertarian/conservative/totalitarian/center")
        music_genre: str = Field(..., description="If topic is music, return genre")
        sport: str = Field(..., description="If topic is sport, return: 'type, club, country'")
        
        
    def analyze_profiles(self, profiles):
        analyzed = []
        for profile in profiles:
            screen_name, name, description = profile
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are an AI that takes Twitter profiles 'screen name | name | description' and returns who it is."},
                    {"role": "user", "content": f"@{screen_name} | {name} | {description}"}
                ],
                response_format=self.AnalysisProfileName,
                temperature=1
            )
            output = json.loads(response.choices[0].message.parsed.json())
            output["screen_name"] = screen_name
            analyzed.append(output)
            print(output)
        return analyzed

class GPT4o():
    client = openai.OpenAI(api_key="

############################## TWEET ANALYSIS ####################################
    class Sport(BaseModel):
        sport: str = Field(..., description="Sport type (football, basketball, ...)")
        sentiment: Optional[str] = Field(..., description="Sentiment about the sport (positive/negative/neutral), prefer neutral if there is no explicit opinion on sport itself")

    class Club(BaseModel):
        club: str = Field(..., description="Official name of mentioned club, not nickname")
        sport: str = Field(..., description="Sport type (football, basketball, ...)")
        sentiment: str = Field(..., description="Sentiment about the club (positive/negative/neutral)")
        country: str = Field(..., description="ISO 3166 code of country of origin")

    class Player(BaseModel):
        player: str = Field(..., description="Player name")
        sport: str = Field(..., description="Sport type (football, basketball, ...)")
        sentiment: str = Field(..., description="Sentiment about the player (positive/negative/neutral)")
        country: str = Field(..., description="ISO 3166 code of country of origin")

    class Genre(BaseModel):
        genre: str = Field(..., description="Mentioned genre")
        sentiment: str = Field(..., description="Sentiment about the genre")

    class Artist(BaseModel):
        artist: str = Field(..., description="Name of artist")
        genre: str = Field(..., description="Genre of the artist")
        sentiment: str = Field(..., description="Sentiment about the artist")
        country: str = Field(..., description="ISO 3166 code of country of origin")
     
        
    class PoliticsModel(BaseModel):
        ideology: str = Field(..., description="Choose ideology of author of the tweet: liberalism, nationalism, conservatism, socialism, communism, environmentalism, social democracy, progressivism, anarchism, centrism, libertarianism, fascism, authoritarianism, religious-based ideology")
        #sentiment: str = Field(..., description="Describe relation of author to mentioned ideology: positive, neutral, negative, critical")
        
    class SportModel(BaseModel):
        sports: list["Sport"]
        clubs: Optional[list["Club"]]
        players: Optional[list["Player"]] = Field(..., description="Write names of players you recognised/known, verify that analysed value is a real player")

    class MusicModel(BaseModel):
        genres: Optional[list["Genre"]] = Field(..., description="Music genres and sentiments")
        artists: Optional[list["Artist"]] = Field(..., description="Artist names and sentiments")

    class OtherTopic(BaseModel):
        topic: str = Field(..., description="Choose from following based on present entities: Finance, Entertainment, Education, Technology, Science, Health, Art and Culture, Hobbies, Nature, Other")
        specification_of_topic: str = Field(..., description="Subdomain of mentioned topic (example: if topic is finance -> real estate/bussiness/crypto/...)")
        
        
    class AnalysisTweet(BaseModel):
        type: str = Field(..., description="Type of analysis ('politics'/'sport'/'music'/'other')")
        language: str = Field(..., description="ISO 639 code of detected language")
        #professionality: float = Field(..., description="Professionality score (between 0 and 1)")
        politics: str = Field(..., description="Choose ideology of author of the tweet: liberalism, nationalism, conservatism, socialism, communism, environmentalism, social democracy, progressivism, anarchism, centrism, libertarianism, fascism, authoritarianism, religious-based ideology")  # Provide default values
        sport: Optional["SportModel"] = Field(..., description="Describe mentioned sports, players and clubs and analyse sentiment of each; notes: it should consider, that sentiment can change throughout the tweet (ex.: one club mentioned positively and other negatively; sport sentiment=positive, club sentiment=negative; etc), dislike for club doesnt mean dislike for sport") #, try also recognise famous sport chants and symbols
        music: Optional["MusicModel"] = Field(..., description="Describe mentioned music genres and musicians")
        #other_topics: List["OtherTopic"] = Field(default_factory=dict, description="Other topics")
        other_topics: Optional[List["str"]] = Field(..., description="Choose primary topic of the tweet if possible, choose from (Finance, Entertainment, Education, Technology, Science, Health, Art and Culture, Hobbies, Nature, Other)")
        
    class AnalysisReaction(BaseModel):
        #original_tweet: "AnalysisTweet" = Field(..., description="Analysis  of original tweet")
        reaction: "AnalysisTweet" = Field(..., description="Analyse reaction if possible")
        reaction_sentiment: str = Field(..., description="Agreeing, Disagreeing, Neutral")

    ### ANALYZA OBSAHU TEXTU V TWEETE ###
    def analyze_tweet(self, text):
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an AI that analyzes content of following tweets and returns structured JSON output."},
                {"role": "user", "content": f"{text}"}
            ],
            response_format=self.AnalysisTweet,
            temperature=0.85
        )
    
        return json.loads(response.choices[0].message.parsed.json())

    def analyze_reaction(self, reaction, original_tweet):
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an AI that analyzes content of twitter comments/quotes and original tweets and returns structured JSON output."},
                {"role": "user", "content": (
                    f"### Original Tweet ###\n"
                    f"{original_tweet}\n\n"
                    f"### Reaction ###\n"
                    f"{reaction}\n\n"
                    f"Analyze reaction and how it relates to the original tweet. Does it agree, disagree, add context, or shift the sentiment?"
                )}
            ],
            response_format=self.AnalysisReaction,
            temperature=0.85
        )
    
        return json.loads(response.choices[0].message.parsed.json())


############################ PROFILE ANALYSIS #######################################
    
    class AnalysisProfileName(BaseModel):
        type: str = Field(..., description="Choose from following words to describe twitter profile: celebrity/fanpage/company/amusement-education/private profile")
        topic: str = Field(...,description="politics/music/sport/other")
        short_describtion: str = Field(...,description="Describe who it is")
        country: str = Field(...,description="Sphere of influence (example: if musician is born in russia but sing in france and in french, return 'france')")
        political_orientation: str = Field(..., description="If topic is politics, choose from: liberalism, nationalism, conservatism, socialism, communism, environmentalism, social democracy, progressivism, anarchism, centrism, libertarianism, fascism, authoritarianism, religious-based ideology")
        music_genre: str = Field(..., description="If topic is music, return genre")
        sport: str = Field(..., description="If topic is sport, return: 'type, club, country'")

    ### ANALYZA DAT IBA NA ZAKLADE TWITTER PROFILU A VSEOBECNYCH ZNALOSTI ###   
    def analyze_profiles(self, profiles):
        analyzed = []
        for profile in profiles:
            screen_name, name, description = profile
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an AI that takes Twitter profiles 'screen name | name | description' and returns who it is."},
                    {"role": "user", "content": f"@{screen_name} | {name} | {description}"}
                ],
                response_format=self.AnalysisProfileName,
                temperature=1
            )
            output = json.loads(response.choices[0].message.parsed.json())
            output["screen_name"] = screen_name
            analyzed.append(output)
            print(output)
        return analyzed
    

    ### ANALYZA VYUZIVAJUCA GOOGLE SEARCH API DATA ###
    def analyze_profile_II(self, data):
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an AI that categorizes profiles into structured data."},
                {"role": "user", "content": f'{data}'}
            ],
            response_format=TopicData,
            temperature=1
        )

        output = response.choices[0].message.parsed.json()
        output = json.loads(output)
        return output

    def profiles_summary(self, data):
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an AI that summarises data into predefined structure."},
                {"role": "user", "content": f'{data}'}
            ],
            response_format=Interests,
            temperature=1
        )

        output = response.choices[0].message.parsed.json()
        return json.loads(output)


    
################## GENERALISATION OF OTHER TOPICS ################################
    def generalise(self, data):
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an AI that takes dictionary and merges keys with similar topic into one. (ex. {Bitcoin : 1}, {Ethereum : 1} -> {Cryptocurrency : 2})"},
                {"role": "user", "content": f'{data}'}
            ],
            response_format=OtherTopics,
            temperature=1
        )

        output = response.choices[0].message.parsed.json()
        return json.loads(output)

###################### NAME SUBBUBBLES ###############################
    class SubgroupName(BaseModel):
        sumarizing_name: str = Field(..., description="Return name of one or multiple topics/entities, which are mentioned in edges the most")
    
    def name_subbubble(self, data):
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an AI that receives set of edges, which contain 4 values: political ideology(True when similar/False when different), sport(set of entities common for both nodes), music(set of entities common for both nodes), other(set of other topics common for both nodes). Your task is to return value of topic/s, which define group of edges the best"},
                {"role": "user", "content": f'{data}'}
            ],
            response_format=self.SubgroupName,
            temperature=1
        )

        output = response.choices[0].message.parsed.json()
        return json.loads(output)
    




class SerpAPI():
    API_KEY = ""#""

    def get_entity(self, name):
        params = {
            "q": name,
            "engine": "google",
            "api_key": self.API_KEY
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        print(data)
        return data
    
    def process_entity(self, entity_data):
        formated_return = {
            "Name": "",
            "Type": "",
            "Entity": ""
        }
        sport = entity_data.get('sports_results', False)
        if sport:
            formated_return["Name"] = sport.get("title", "N/A")
            formated_return["Type"] = sport.get("team", "N/A")
            formated_return["Clubs"] = []
            for i in sport.get("tables", []):
                formated_return["Clubs"].append(i.get("title", "N/A"))

        kg = entity_data.get('knowledge_graph', False)
        if kg:
            formated_return["Name"] = kg.get("title", "N/A")
            formated_return["Type"] = kg.get("type", "N/A")
            formated_return["Entity"] = kg.get("entity_type", "N/A")
        return formated_return

class GSE:

    API_KEY = "
    SEARCH_ENGINE_ID = ""
    
    def get_entity(self, name):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.API_KEY,
            'cx': self.SEARCH_ENGINE_ID,
            'q': name
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json()
            with open("vysledky.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            rtrn = []
            for i in range(min(2, len(results["items"]))):
                rtrn.append(results["items"][i].get("snippet", ""))
            return rtrn
        else:
            print(f"Error: {response.status_code}, {response.text}")

    def process_entity(self, entity_data):
        return {"potential describtion data": entity_data}

