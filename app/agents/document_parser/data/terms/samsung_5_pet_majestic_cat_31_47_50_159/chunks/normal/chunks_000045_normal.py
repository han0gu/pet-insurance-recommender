from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[최저보증이율]\n'
 '운용자산이익률 및 시중금리가 하락하더라도 회사에서 보증하는 최저한도의 적용이율입니다. 예를 들어, 적립금이 공시이율에 따라 부리되며 '
 '공시이율이 0.1%인 경우(최저보증이율이 공시이율보다 큰 경우), 적립금은 공시이율(0.1%)이 아닌 최저보증이율로 부리됩니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 35},
 'term_type': 'basic',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000045',
              'chunk_char_len': 161,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
