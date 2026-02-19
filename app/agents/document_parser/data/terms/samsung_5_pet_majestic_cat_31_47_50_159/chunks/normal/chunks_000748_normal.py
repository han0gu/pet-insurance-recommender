from langchain_core.documents import Document

chunk = Document(
    page_content=('. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관 제27조 (특별약관의 재 가입에 관한사항) 제5항에 따라 보험계약이 연장된 '
 '경우에는 보장개시일(책임개시일) 은 이 특별약관의 보험계약일로 봅니다. ⑨ 제4항의 보상한도액에도 불구하고 동일한 날에 아래의 특별약관 '
 '중 피보험자가 가입'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 117},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000748',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
