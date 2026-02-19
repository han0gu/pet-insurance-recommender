from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 평균공시이율: 전체 보험회사 공시이율의 평균으로, 이 계약 체결 시점의 이율을 말합니다. 이 평균공시이율은 금융감독원 '
 '홈페이지(www.fss.or.kr)의 「업무자료/'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000530',
              'chunk_char_len': 96,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
