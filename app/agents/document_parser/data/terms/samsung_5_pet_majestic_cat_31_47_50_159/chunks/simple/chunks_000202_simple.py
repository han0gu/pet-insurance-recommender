from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 제2항에 의하여 추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에 따라\n'
 '회사가 추정하는 보험금의 50% 상당액을 가지급보험금으로 지급합니다.\n'
 '<용어풀이>\n'
 '[가지급보험금]\n'
 '보험금 지급이 늦어지는 경우 보험수익자 청구에 따라 확정된 보험금을 먼저 지급하는 제도'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 52},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000202',
              'chunk_char_len': 151,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
