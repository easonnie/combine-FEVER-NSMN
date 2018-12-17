class DocRetriver(object):
    def add_doc(self, doc):
        """
        Add a given document to this collection
        :param doc:
        :return:
        """
        pass

    def query(self, query, n_max=1):
        """
        Query the collection by the specific query.

        Note: Query and doc should have similar structure
            We will obtain feature vector using the following method

            doc_vector = self._build_doc_feature_vec(doc)
            query_vector = self._build_index_feature_vec(query)

            Then compare the query vector with all the doc_vector.

            Similarity_Score(doc_vector, query_vector)


        :param query:
        :param n_max:
        :return:
        """
        pass

    def _build_doc_feature_vec(self, tokens):
        pass

    def _build_index_feature_vec(self, tokens):
        pass

