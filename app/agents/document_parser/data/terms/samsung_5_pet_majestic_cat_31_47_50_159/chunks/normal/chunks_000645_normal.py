from langchain_core.documents import Document

chunk = Document(
    page_content='4-2. 반려묘 수술비(치과및구강질환포함) 확대보장(재가입형) 추가특별약관\n제1조 (보험금의 지급사유)',
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 107},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000645',
              'chunk_char_len': 57,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
