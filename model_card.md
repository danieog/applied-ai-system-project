# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **Music Controller**  

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

This reccomender is for people who don't have time to sit and discover new music. This assumes the user's prefer to listen to music from their favorite genre. This is for classroom exploration, since most real songs don't have actual energy percentages.
---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

This takes in your favorite genre, mood, your target energy, enjoyment of acoustics, and your likes & skips. Then, we calculate a number and use that to determine tyour favorite song. We consider everything, though they're all weighed differently. 30% of your score is from the genre, 25% from energy, 20% from the mood, 15% for acoustics, and 10% for feedback.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  
There are 20 songs total in the catalog. The genres represented are pop, lofi, rock, ambient, jazz, synthwave, indie pop, electronic, country, classical, reggae, metal, folk, and blues. The moods are uplifting, melancholic, adventurous, peaceful, angry, joyful, romantic, confident, sad, energetic, happy, focused, moody, relaxed, chill and intense. I only added more songs to expand the data set. I don't think anything is missing.
---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  
My system gives the best results for users who like a specific genre (and they can match the rest of the results to their preferences). The patterns mainly consist of matching slow energy songs to their moods (and vice versa). 
---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  
This system does not consider danceability, valence, or tempo. Some genres that maybe under represented are everything besides lofi and underpresented moods are everythign besides chill. The scoring also favors users who like genres with high energy.
---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  

No need for numeric metrics unless you created some.
I tested a generic case, edge cases (four) and one case of me. I looked for the genre first, then the energy. What surprised me was how much genre plays a role into the recommnedation. For my basic case, I wanted someone who like electronic music with high energy. For my first edge case, I wanted the maximum energy with electronic music. My second edge case was as low energy as it got. My third case was having no favorite genre or mood. My fourth edge case had some likes and dislikes. My fifth case was my own tastes that had songs that were indie pop and chill, but with high enegry.
---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  
I would add more diverse songs to test with, along with a more concrete scoring plan. I also will try to account for tempo, valence and danceability.
---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  
I learned that recommender systems are algorithm based and that those algorithms need to be perfect before we implement them. Something interesting I discovered was how much math is necessary for this. This changed the way I think about music apps, since now I'm going to look into this more, rather than just thinking it gave me similar artists.