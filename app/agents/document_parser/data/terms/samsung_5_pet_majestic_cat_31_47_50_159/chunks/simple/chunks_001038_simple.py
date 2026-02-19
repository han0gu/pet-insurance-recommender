from langchain_core.documents import Document

chunk = Document(
    page_content=('주 ) 1. 향후 「감염병의 예방 및 관리에 관한 법률」이 개정되어 “감염병병원체 확인기관”의 내용이 변경된 경우, 변경된 내용을 '
 '적용합니다. 2 . 향후 「감염병의 예방 및 관리에 관한 법률」등 관계법령에서 제외되는 감염병이 생기는 경우 해당 감염병은 신고여부와 '
 '상관없이 의사의 진단에 따릅니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 159},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001038',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
