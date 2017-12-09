from sklearn.feature_extraction import DictVectorizer


class SimpleVectorizer(DictVectorizer):
    def __init__(self, field):
        super(SimpleVectorizer, self).__init__()
        self.field = field

    def fit_transform(self, X, y=None):
        return super(SimpleVectorizer, self).fit_transform([{self.field: getattr(d, self.field)} for d in X])
