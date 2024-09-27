from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class LDA():
    def __init__(self, docs):
        # Преобразовать документ в векторы TF
        self.TF = CountVectorizer()
        self.TF.fit(docs)
        vectors = self.TF.transform(docs)

        # Построить тематическую модель LDA
        self.LDA_model = LatentDirichletAllocation(n_components=5)
        self.LDA_model.fit(vectors)
        return

    def get_features(self, new_docs):
        # Получить тематические признаки для новых документов
        new_vectors = self.TF.transform(new_docs)
        return self.LDA_model.transform(new_vectors)

if __name__ == '__main__':
    # Позже, на этапе эксплуатации, создать экземпляр модели LDA
    docs = ['This is a text.', 'This another one.']
    LDA_featurizer = LDA(docs)

    # Получить тематические признаки для new_docs
    new_docs = ['This is a third text.', 'This is a fourth one.']
    LDA_features = LDA_featurizer.get_features(new_docs)
    print(LDA_features)