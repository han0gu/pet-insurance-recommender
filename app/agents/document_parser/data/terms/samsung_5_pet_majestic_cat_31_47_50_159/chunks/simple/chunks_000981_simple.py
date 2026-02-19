from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다) 치매의 장해평가는 전문의(정신건강의학과, 신경과)에 의한 임상치매척도 (한국판 Expanded Clinical Dementia '
 'Rating) 검사결과에 따른다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 148},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000981',
              'chunk_char_len': 94,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
