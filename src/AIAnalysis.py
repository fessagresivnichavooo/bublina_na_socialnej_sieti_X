from pydantic import BaseModel, Field
from typing import Dict, Optional
import openai
import json


class GPT4o_mini():
    client = openai.OpenAI(api_key="")

    class AnalysisProfileName(BaseModel):
        type: str = Field(..., description="Describes twitter profile based on its name: celebrity/fanpage/company/amusement-education/private profile")
        topic: str = Field(...,description="politics/music/sport/other")
        short_describtion: str = Field(...,description="Describe who it is or write 'dont know'")
        country: str = Field(...,description="Sphere of influence (example: if musician is born in russia but sing in france and in french, return 'france')")
        celebrity_info: str = Field(..., description="Only when type is celebrity or fanpage. If politicly related, return from: liberal/libertarian/conservative/totalitarian/center (must be 1 or 2 of these) ; if music related, return: genre/s; if sport related, return: 'type of sport, sports club(can be null), country(can be null)'")
        
    def analyze_profiles(self, profiles):
        analyzed = []
        for profile in profiles:
            screen_name, name, description = profile
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",#"gpt-4o-2024-08-06",
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


    #print(analyze_profile("@IvanKmotrik").json())
