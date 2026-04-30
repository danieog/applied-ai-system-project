import os
from typing import List, Dict
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    acousticness: float

@dataclass
class UserProfile:
    genre: str
    mood: str
    energy: float
    acousticness: float
    previously_liked: List[str] = None

class RAGRecommender:
    def __init__(self):
        pass

    def generate_recommendation(self, user: UserProfile, top_songs: List[Song]) -> str:
        """
        Generate a natural language recommendation based on user preferences.
        """
        # Select top 3 songs
        recommendations = top_songs[:3]
        
        response = "Based on your preferences, here are my top 3 recommendations:\n\n"
        
        for i, song in enumerate(recommendations, 1):
            explanation = self._generate_explanation(user, song)
            response += f"{i}. {song.title} by {song.artist} - {explanation}\n\n"
        
        return response.strip()

    def _build_prompt(self, user: UserProfile, top_songs: List[Song]) -> str:
        """
        Build the prompt for Claude.
        """
        user_info = f"""
User Profile:
- Genre: {user.genre}
- Mood: {user.mood}
- Energy: {user.energy}/10
- Acousticness: {user.acousticness}/10
"""

        if user.previously_liked:
            user_info += f"- Previously liked: {', '.join(user.previously_liked)}\n"

        songs_info = "\nTop Songs (ranked by score):\n"
        for i, song in enumerate(top_songs, 1):
            songs_info += f"{i}. {song.title} by {song.artist} (Genre: {song.genre}, Mood: {song.mood}, Energy: {song.energy*10:.1f}/10, Acousticness: {song.acousticness*10:.1f}/10)\n"

        prompt = f"""
{user_info}
{songs_info}

Based on the user's profile, recommend the top 3 songs from the list above. For each recommendation, provide:
1. The song title and artist
2. A 1-2 sentence explanation of why it fits the user's taste

Only recommend songs from the provided list. Be conversational and natural.
"""

        return prompt