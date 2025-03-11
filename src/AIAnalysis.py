from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import openai
import json
import requests


############# PYDANTIC #######################

# === Sport Models ===
class Athlete(BaseModel):
    type_of_sport: str = Field(..., description="Type of sport (e.g., Football, Basketball)")
    club: Optional[str] = Field(None, description="Club/team the athlete plays for")
    nationality: Optional[str] = Field(None, description="Nationality of the athlete")


class ClubTeam(BaseModel):
    sport: str = Field(..., description="Type of sport")
    country: Optional[str] = Field(None, description="Country where the club/team is based")


# === Music Models ===
class Author(BaseModel):
    genre: str = Field(..., description="Music genre of the author")
    country: Optional[str] = Field(None, description="Country where the author is influential")


class BandLabel(BaseModel):
    music_genre: str = Field(..., description="Music genre of the band/label")
    authors: Optional[List[str]] = Field(None, description="List of authors in the band/label")
    country: Optional[str] = Field(None, description="Country of influence")


class Fanpage(BaseModel):
    genre_of_music: str = Field(..., description="Music genre related to the fanpage")


# === Politics Models ===
class Politician(BaseModel):
    political_party: Optional[str] = Field(None, description="Political party the politician belongs to")
    ideology: Optional[str] = Field(None, description="Political ideology (liberal, conservative, etc.)")
    country: str = Field(..., description="Country where the politician is active")


class PoliticalPartyMovement(BaseModel):
    ideology: str = Field(..., description="Political ideology (e.g., Conservative, Liberal, Socialist)")
    country: str = Field(..., description="Country where the political party/movement is active")    


class JournalistNews(BaseModel):
    country: str = Field(..., description="Country where the journalist/news is active")
    ideology: Optional[str] = Field(None, description="Political ideology of the journalist/news")

# === Other Models ===

class Finance(BaseModel):
    sector: str = Field(..., description="Specific sector in finance (e.g., banking, investment, cryptocurrency)")
    country: Optional[str] = Field(None, description="Country where the financial organization or expert operates")


class Entertainment(BaseModel):
    industry: str = Field(..., description="Entertainment industry category (e.g., movies, gaming, television)")
    notable_work: Optional[str] = Field(None, description="Notable work, platform, or project related to entertainment")


class Technology(BaseModel):
    field: str = Field(..., description="Specific field in technology (e.g., AI, robotics, software, hardware)")
    notable_company: Optional[str] = Field(None, description="Major company or project associated with this entity")


class Education(BaseModel):
    specialization: str = Field(..., description="Field of education (e.g., STEM, humanities, business)")
    institution: Optional[str] = Field(None, description="Associated school, university, or institution")



# === Unified Topic Model ===
class TopicData(BaseModel):
    topic: str = Field(..., description="Main category (Sport, Music, Politics, Other)")
    type: str = Field(..., description="Specific entity type within the topic")
    data: Union[
        Athlete,
        ClubTeam,
        Author,
        BandLabel,
        Fanpage,
        Politician,
        PoliticalPartyMovement,
        JournalistNews,
        Finance,
        Entertainment,
        Technology,
        Education
    ]


###############################################################################






class GPT4o_mini():
    client = openai.OpenAI(api_key="")

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
    client = openai.OpenAI(api_key="")

############################## TWEET ANALYSIS ####################################
    class Sport(BaseModel):
        sport: str = Field(..., description="Sport type (football, basketball, ...)")
        sentiment: Optional[str] = Field(..., description="Sentiment about the sport (positive/negative/neutral), prefer neutral if there is no explicit opinion on sport itself")

    class Club(BaseModel):
        club: str = Field(..., description="Official name of mentioned club, not nickname")
        sentiment: str = Field(..., description="Sentiment about the club (positive/negative/neutral)")

    class Player(BaseModel):
        player: str = Field(..., description="Player name")
        sentiment: str = Field(..., description="Sentiment about the player (positive/negative/neutral)")

    class Genre(BaseModel):
        genre: str = Field(..., description="Mentioned genre")
        sentiment: str = Field(..., description="Sentiment about the genre")

    class Artist(BaseModel):
        artist: str = Field(..., description="Name of artist")
        genre: str = Field(..., description="Genre of the artist")
        sentiment: str = Field(..., description="Sentiment about the artist")
     
        
    class PoliticsModel(BaseModel):
        E: float = Field(..., description="Economic freedom score (-1 to 1), negative number = state controlled , positive number = free market, individualism")
        S: float = Field(..., description="Social freedom score (-1 to 1); negative number is favoring government control; positive is against government control, more individual freedom")
        Reason: str = Field(..., description="Explain for both E and S why they got assigned these values")

    class SportModel(BaseModel):
        sports: list["Sport"]
        clubs: Optional[list["Club"]]
        players: Optional[list["Player"]] = Field(..., description="Write names of players you recognised/known, verify that analysed value is a real player")

    class MusicModel(BaseModel):
        genres: Optional[list["Genre"]] = Field(..., description="Music genres and sentiments")
        artists: Optional[list["Artist"]] = Field(..., description="Artist names and sentiments")

    class AnalysisTweet(BaseModel):
        type: str = Field(..., description="Type of analysis ('politics'/'sport'/'music'/'other')")
        language: str = Field(..., description="Detected language")
        professionality: float = Field(..., description="Professionality score (between 0 and 1)")
        politics: Optional["PoliticsModel"] = Field(..., description="Describes vector on Nolans chart represented by political ideology in current tweet; conservative/right-wing = +E -S , liberal/left-wing = +S -E , authoritarian = -S -E , libertarian = +S +E")  # Provide default values
        sport: Optional["SportModel"] = Field(..., description="Describe mentioned sports, players and clubs and analyse sentiment of each; notes: it should consider, that sentiment can change throughout the tweet (ex.: one club mentioned positively and other negatively; sport sentiment=positive, club sentiment=negative; etc), dislike for club doesnt mean dislike for sport") #, try also recognise famous sport chants and symbols
        music: Optional["MusicModel"] = Field(..., description="Describe mentioned music genres and musicians")
    ##    other_topics: Dict[str, str] = Field(default_factory=dict, description="Other topic sentiments")

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


############################ PROFILE ANALYSIS #######################################
    
    class AnalysisProfileName(BaseModel):
        type: str = Field(..., description="Choose from following words to describe twitter profile: celebrity/fanpage/company/amusement-education/private profile")
        topic: str = Field(...,description="politics/music/sport/other")
        short_describtion: str = Field(...,description="Describe who it is")
        country: str = Field(...,description="Sphere of influence (example: if musician is born in russia but sing in france and in french, return 'france')")
        political_orientation: str = Field(..., description="If topic is politics, choose from: liberal/libertarian/conservative/totalitarian/center")
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
        return output
    

    




class SerpAPI():
    API_KEY = ""

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




##a = SerpAPI()
##print(a.process_entity(a.get_entity("Tigran Barseghyan")))


##for i in ["Alex Jones"]:
##    c = a.get_entity(i)
##    print(c)
##    b = a.process_entity(c)
##    print(b)
##    print(GPT4o().analyze_profile_II(b))




'''
1. Liberalism
Classical Liberalism – Individual rights, free markets, limited government
Social Liberalism – Welfare state, regulated capitalism
Libertarianism – Minimal government, strong personal freedoms
2. Conservatism
Traditional Conservatism – Stability, tradition, gradual change
Fiscal Conservatism – Low taxes, small government
Social Conservatism – Religious and family values
3. Socialism
Democratic Socialism – Social justice, regulated capitalism
Marxism – Class struggle, worker control of production
4. Communism
Marxist-Leninism – One-party rule, planned economy
Maoism – Peasant-based revolutionary socialism
5. Fascism & Authoritarianism
Fascism – Nationalism, authoritarian control, militarism
Nazism – Racial supremacy, totalitarianism
6. Nationalism
Civic Nationalism – Nation based on shared values/culture
Ethnic Nationalism – Nation based on ancestry/heritage
7. Anarchism
Anarcho-Communism – Stateless, classless society
Anarcho-Capitalism – Stateless free markets
8. Environmentalism
Green Politics – Sustainability, climate action
9. Religious-Based Ideologies
Theocracy – Religious rule over government
Christian Democracy – Conservative social values with welfare policies
Islamism – Political enforcement of Islamic law
10. Centrism & Pragmatic Approaches
Centrism – Balance between left and right policies
Third Way – Mix of capitalism and social democracy
'''

### vyskusat pristup: vypisat ideologie, a pre kazdy tweet napisat, ako velmi pre alebo proti
### ak profil, tak vyberie iba hlavnu temu a nie podtemy (cize conservativism, liberalism ...)
### neskor mozno vyskusat priradit kazdej ideologii vektor a podla toho umiestnit na spektrum
















