from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class LSA():
    def __init__(self, docs):
        # Преобразовать документ в векторы TF-IDF
        self.TF_IDF = TfidfVectorizer()
        self.TF_IDF.fit(docs)
        vectors = self.TF_IDF.transform(docs)

        # Построить тематическую модель LSA
        self.LSA_model = TruncatedSVD(n_components=5)
        self.LSA_model.fit(vectors)
        return

    def get_features(self, new_docs):
        # Получить тематические признаки для новых документов
        new_vectors = self.TF_IDF.transform(new_docs)
        return self.LSA_model.transform(new_vectors)

if __name__ == '__main__':
    # Позже, на этапе эксплуатации, создать экземпляр модели LSA
    docs = ['This is a text.', 'This another one.']
    LSA_featurizer = LSA(docs)

    # Получить тематические признаки для new_docs
    new_docs = ['This is a third text.', 'This is a fourth one.']
    LSA_features = LSA_featurizer.get_features(new_docs)
    print(LSA_features)