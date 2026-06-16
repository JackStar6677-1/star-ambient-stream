import pickle
import os
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

TOKEN_PICKLE_FILE = "token.pickle"

def main():
    if not os.path.exists(TOKEN_PICKLE_FILE):
        print("Error: token.pickle not found.")
        return
        
    with open(TOKEN_PICKLE_FILE, 'rb') as token:
        creds = pickle.load(token)
        
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        
    youtube = build('youtube', 'v3', credentials=creds)
    
    print("--- ACTIVE BROADCASTS ---")
    broadcasts = youtube.liveBroadcasts().list(
        part='id,snippet,status,contentDetails',
        broadcastStatus='all',
        maxResults=10
    ).execute()
    
    for b in broadcasts.get('items', []):
        print(f"ID: {b['id']}")
        print(f"  Title: {b['snippet']['title']}")
        print(f"  Status: {b['status']['lifeCycleStatus']}")
        print(f"  Privacy: {b['status']['privacyStatus']}")
        print(f"  Bound Stream: {b['contentDetails'].get('boundStreamId')}")
        print("-" * 30)
        
    print("\n--- ACTIVE STREAMS ---")
    bound_stream_ids = [b['contentDetails'].get('boundStreamId') for b in broadcasts.get('items', []) if b['contentDetails'].get('boundStreamId')]
    if bound_stream_ids:
        streams = youtube.liveStreams().list(
            part='id,snippet,status,cdn',
            id=bound_stream_ids[0]
        ).execute()
    else:
        streams = {'items': []}
    
    for s in streams.get('items', []):
        print(f"ID: {s['id']}")
        print(f"  Title: {s['snippet']['title']}")
        print(f"  Status: {s['status']['streamStatus']}")
        print(f"  Health: {s['status'].get('healthStatus', {}).get('status')}")
        print(f"  Health Details: {s['status'].get('healthStatus')}")
        print("-" * 30)

if __name__ == '__main__':
    main()
