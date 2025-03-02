from pydantic import BaseModel, Field
from typing import Dict, Optional
import json

class Sport(BaseModel):
    sport: str = Field(..., description="Sport type (football, basketball, ...)")
    sentiment: Optional[str] = Field(..., description="Sentiment about the sport (positive/negative/neutral), if there is not specific sentiment/opinion about sport itself, keep it neutral")

class Club(BaseModel):
    club: str = Field(..., description="Official name of mentioned club, not nickname")
    sentiment: str = Field(..., description="Sentiment about the club (positive/negative/neutral)")

class Player(BaseModel):
    player: str = Field(..., description="Player name")
    sentiment: str = Field(..., description="Sentiment about the player (positive/negative/neutral)")

 
    
class PoliticsModel(BaseModel):
    E: float = Field(..., description="Economic freedom score (-1 to 1), higher value = less controled economic")
    S: float = Field(..., description="Social freedom score (-1 to 1), higher value = more individual freedom")
    Reason: str = Field(..., description="Reason for choice of values for E and S")

class SportModel(BaseModel):
    sports: list[Sport]
    clubs: Optional[list[Club]]
    players: Optional[list[Player]] = Field(..., description="Write names of players you recognised/known, verify that analysed value is a real player")

class MusicModel(BaseModel):
    genres: Dict[str, str] = Field(..., description="Music genres and sentiments")
    artists: Dict[str, str] = Field(..., description="Artist names and sentiments")

class AnalysisResult(BaseModel):
    type: str = Field(..., description="Type of analysis ('politics'/'sport'/'music'/'other')")
    language: str = Field(..., description="Detected language")
    professionality: float = Field(..., description="Professionality score (between 0 and 1)")
##    how_much_political: float = Field(..., description="Political intensity (0-1)")
    politics: Optional[PoliticsModel] = Field(..., description="Describes vector on Nolans chart represented by tweets political idea, IMPORTANT: ITS SUPPOSED TO ANALYSE POLITICAL VIEW OF AUTHOR OF TWEET, NOT POLITICAL VIEWS MENTIONED(for example, if author express critisism against totalitarian regime, S should be positive)")  # Provide default values
    sport: Optional[SportModel] = Field(..., description="Describes meantioned sports, players and clubs and analysis sentiment of each; notes: it should consider, that sentiment can change throughout the tweet (ex.: one club mentioned positively and other negatively; sport sentiment=positive, club sentiment=negative; etc), dislike for club doesnt mean dislike for sport") #, try also recognise famous sport chants and symbols
##    music: Optional[MusicModel]# = Field(default_factory=MusicModel)
##    other_topics: Dict[str, str] = Field(default_factory=dict, description="Other topic sentiments")
  
'----------------------------------------------------------------------------------'

import openai

client = openai.OpenAI(api_key="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")

def analyze_text(text: str) -> AnalysisResult:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are an AI that analyzes content of following tweets and returns structured JSON output."},
            {"role": "user", "content": f"{text}"}
        ],
        response_format=AnalysisResult,
        temperature=1
    )
    
    return response.choices[0].message.parsed


####text = """
####AEK ❌
####"""
####result = analyze_text(text)
####print(result.json())




##example_data = {
##    "type": "politics",
##    "language": "en",
##    "professionality": 0.8,
##    "how_much_political": 0.7,
##    "politics": {"E": 0.5, "S": -0.2},
##    "sport": {"sports": {"football": "positive"}, "clubs": {"FC Barcelona": "neutral"}, "players": {"Messi": "positive"}},
##    "music": {"genres": {"rock": "positive"}, "artists": {"Queen": "neutral"}},
##    "other_topics": {"technology": "positive"}
##}
##result = AnalysisResult(**example_data)
##print(result.model_dump_json(indent=4))



'___________________________________________________________________________________________________________________________________________________________________________________'

class AnalysisProfileName(BaseModel):
    type: str = Field(..., description="Describes twitter profile based on its name: celebrity/fanpage/company/amusement-education/private profile")
    topic: str = Field(...,description="politics/music/sport/other")
    short_describtion: str = Field(...,description="Describe who it is or write 'dont know'")
    country: str = Field(...,description="Sphere of influence (example: if musician is born in russia but sing in france and in french, return 'france')")
    celebrity_info: str = Field(..., description="Only when type is celebrity or fanpage. If politicly related, return: liberal/libertarian/conservative/totalitarian/center (must be 1 or 2 of these) ; if music related, return: genre/s; if sport related, return: 'type of sport, sports club(can be null), country(can be null)'")
    
def analyze_profiles(profiles):
        analyzed = []
        for profile in profiles:
            screen_name, name, description = profile
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",#"gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an AI that takes Twitter profiles 'screen name | name | description' and returns who it is."},
                    {"role": "user", "content": f"@{screen_name} | {name} | {description}"}
                ],
                response_format=AnalysisProfileName,
                temperature=1
            )
            output = json.loads(response.choices[0].message.parsed.json())
            output["screen_name"] = screen_name
            analyzed.append(output)
        return analyzed


data = [("realDonaldTrump", "Donald J. Trump", "45th & 47th President of the United States of America"),("vladimir_weiss", "Vladimir Weiss", "The official twitter account of vladimir, professional football player at Lekhwiya SC.")]
profiles = {}
for i in analyze_profiles(data):
    profiles[i["screen_name"]] = i

with open("profile_analysis.json", "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=4, ensure_ascii=False)


