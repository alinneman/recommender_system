# Example Training Loop
def train(model, dataloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for user_items, user_cats, pos_items, pos_cats, neg_items, neg_cats in dataloader:
            optimizer.zero_grad()
            pos_score, neg_score = model(user_items, user_cats, pos_items, pos_cats, neg_items, neg_cats)
            loss = criterion(pos_score, neg_score)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# FAISS KNN Retrieval
def retrieve_top_k(model, item_embeddings, user_embedding, k=10):
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(item_embeddings)
    _, indices = index.search(user_embedding, k)
    return indices

# Evaluation Metrics
def evaluate(model, dataloader):
    model.eval()
    recall_at_k = {1: 0, 5: 0, 10: 0, 20: 0}
    total_samples = 0
    with torch.no_grad():
        for user_items, user_cats, pos_items, pos_cats, neg_items, neg_cats in dataloader:
            user_emb = model.user_tower(user_items, user_cats)
            pos_emb = model.item_tower(pos_items, pos_cats)
            scores = torch.matmul(user_emb, pos_emb.T)
            for k in recall_at_k.keys():
                top_k = scores.topk(k, dim=1).indices
                recall_at_k[k] += (top_k == 0).sum().item()
            total_samples += user_emb.size(0)
    for k in recall_at_k.keys():
        recall_at_k[k] /= total_samples
    print("Evaluation - Recall@K:", recall_at_k)
    return recall_at_k