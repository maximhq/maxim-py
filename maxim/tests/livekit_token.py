from livekit import api
import os

from dotenv import load_dotenv
load_dotenv()

token = api.AccessToken(os.getenv('LIVEKIT_API_KEY'), os.getenv('LIVEKIT_API_SECRET')) \
    .with_identity("identity") \
    .with_name("name") \
    .with_grants(api.VideoGrants(
        room_join=True,
        room="my-room",
    )).to_jwt()

print(token)
