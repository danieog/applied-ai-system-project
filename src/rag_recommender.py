import os
import logging
from typing import List, Dict
from dataclasses import dataclass, field
from dotenv import load_dotenv
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(
    filename="rag_recommender.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    acousticness: float
    mood_tags: str = ""


@dataclass
class UserProfile:
    genre: str
    mood: str
    energy: float
    acousticness: float
    previously_liked: List[str] = field(default_factory=list)


class RAGRecommender:
    def __init__(self):
        self.groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.Client()
        self.collection = None         # songs index
        self.context_collection = None  # genre/mood knowledge-base index
        self.last_context_docs: List[str] = []  # context retrieved in last generate call

    def build_index(self, songs: List[Dict]) -> None:
        """Embed all songs and store in ChromaDB for semantic retrieval."""
        try:
            self.chroma_client.delete_collection("songs")
        except Exception:
            pass
        self.collection = self.chroma_client.create_collection("songs")

        docs = [self._song_to_text(s) for s in songs]
        embeddings = self.embedder.encode(docs).tolist()
        self.collection.add(
            documents=docs,
            embeddings=embeddings,
            ids=[str(s["id"]) for s in songs],
            metadatas=[{
                "title": s["title"],
                "artist": s["artist"],
                "genre": s["genre"],
                "mood": s["mood"],
                "energy": float(s["energy"]),
                "acousticness": float(s["acousticness"]),
                "mood_tags": s.get("mood_tags", ""),
            } for s in songs],
        )

    def build_context_index(self, docs: List[Dict]) -> None:
        """
        Index genre/mood knowledge-base documents into a separate ChromaDB
        collection so they can be retrieved alongside songs.

        Each doc must have: id (str), content (str), and optionally name/type.
        """
        try:
            self.chroma_client.delete_collection("context")
        except Exception:
            pass
        self.context_collection = self.chroma_client.create_collection("context")

        texts = [d["content"] for d in docs]
        embeddings = self.embedder.encode(texts).tolist()
        self.context_collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=[d["id"] for d in docs],
            metadatas=[{"name": d.get("name", ""), "type": d.get("type", "")} for d in docs],
        )
        logger.info("Context index built with %d documents", len(docs))

    def retrieve_context(self, user: UserProfile, k: int = 2) -> List[str]:
        """
        Query the knowledge-base index with the user profile and return the
        k most relevant document texts (genre/mood descriptions).
        """
        if self.context_collection is None:
            return []
        query = (
            f"Music listener who enjoys {user.genre} with a {user.mood} mood, "
            f"energy {user.energy * 10:.0f}/10."
        )
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.context_collection.query(query_embeddings=query_embedding, n_results=k)
        return results["documents"][0]

    def retrieve(self, user: UserProfile, k: int = 10) -> List[Dict]:
        """Embed the user profile and return the k most semantically similar songs."""
        if self.collection is None:
            return []
        query = (
            f"I enjoy {user.genre} music with a {user.mood} mood. "
            f"Energy: {user.energy * 10:.0f}/10, acousticness: {user.acousticness * 10:.0f}/10."
        )
        if user.previously_liked:
            query += f" Previously liked: {', '.join(user.previously_liked)}."

        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)

        retrieved = []
        for i, meta in enumerate(results["metadatas"][0]):
            retrieved.append({"id": int(results["ids"][0][i]), **meta})
        return retrieved

    def generate_recommendation(self, user: UserProfile, top_songs: List[Song]) -> str:
        """Call Groq (Llama) to produce a natural language explanation for the top songs.

        If a context index has been built, automatically retrieves relevant genre/mood
        knowledge-base documents and injects them into the prompt before calling the LLM.
        """
        context_docs = self.retrieve_context(user, k=2)
        self.last_context_docs = context_docs  # expose for UI display
        prompt = self._build_prompt(user, top_songs[:3], context_docs)
        logger.info(
            "Sending prompt to Groq (%d chars, %d context docs)",
            len(prompt), len(context_docs),
        )
        try:
            response = self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            logger.info("Groq response received (%d chars)", len(text))
            return text
        except Exception as exc:
            logger.error("Groq call failed: %s", exc)
            raise

    def _song_to_text(self, song: Dict) -> str:
        tags = song.get("mood_tags", "")
        return (
            f"{song['title']} by {song['artist']}. "
            f"Genre: {song['genre']}. Mood: {song['mood']}. Tags: {tags}."
        )

    def _build_prompt(
        self,
        user: UserProfile,
        top_songs: List[Song],
        context_docs: List[str] = None,
    ) -> str:
        user_info = (
            f"User Profile:\n"
            f"- Preferred genre: {user.genre}\n"
            f"- Preferred mood: {user.mood}\n"
            f"- Energy level: {user.energy * 10:.0f}/10\n"
            f"- Acousticness: {user.acousticness * 10:.0f}/10\n"
        )
        if user.previously_liked:
            user_info += f"- Previously liked: {', '.join(user.previously_liked)}\n"

        context_section = ""
        if context_docs:
            context_section = "\nMusic Knowledge Base (use this to write specific, informed explanations):\n"
            for doc in context_docs:
                context_section += f"- {doc}\n"

        songs_info = "\nTop matched songs:\n"
        for i, song in enumerate(top_songs, 1):
            songs_info += (
                f"{i}. \"{song.title}\" by {song.artist} "
                f"(Genre: {song.genre}, Mood: {song.mood}, "
                f"Energy: {song.energy * 10:.1f}/10, "
                f"Acousticness: {song.acousticness * 10:.1f}/10)\n"
            )

        return (
            f"{user_info}{context_section}{songs_info}\n"
            "You are a friendly music recommendation assistant. "
            "For each of the 3 songs above, write 1-2 warm, conversational sentences explaining "
            "why it is a great fit for this user's taste. "
            "Draw on the Music Knowledge Base above to make your explanations specific and informed. "
            "Only reference songs from the list."
        )
