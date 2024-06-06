from sentence_transformers import SentenceTransformer, models ,util
import pandas as pd
import torch

try:
    model = SentenceTransformer("khathathorn/modelforreccomendation")
except TypeError as e:
    print("TypeError encountered:", e)
    # Manually load the model components and adjust the pooling configuration
    word_embedding_model = models.Transformer('khathathorn/modelforreccomendation')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Load the data
data = pd.read_csv('deploy/data.csv')

#data = data.drop_duplicates(subset=['menu'], keep='first')

embeddings = torch.load("deploy/embedding.pt", map_location=torch.device('cpu'))

def showmenu(row_id: int) -> str:
    if 0 <= row_id < len(data):
        menu = data['menu'].iloc[row_id]
        return menu
    else:
        return "Invalid row_id"

def findmenu(x: str) -> list:
    query = model.encode([x])
    cosine_scores = util.cos_sim(query, embeddings)
    all_idx = torch.topk(cosine_scores.flatten(), 5).indices
    recommended_menus = []
    for i in all_idx:
        recommended_menus.append(showmenu(i.item()))
    return recommended_menus