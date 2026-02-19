from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 기간과 날짜 관련 용어\n'
 '1. 보험기간: 계약에 따라 보장을 받는 기간을 말합니다. 2. 영업일: 회사가 영업점에서 정상적으로 영업하는 날을 말하며, 토요일, '
 '‘관공서의 공휴일에 관한 규정’에 따른 공휴일과 노동절을 제외합니다.\n'
 '⑤ 보험료 관련 용어\n'
 '1. 보험료 : 손해를 보장하는데 필요한 보험료를 말합니다.\n'
 '⑥ [갱신형] 특별약관의 갱신 관련 용어'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 50},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000184',
              'chunk_char_len': 197,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
