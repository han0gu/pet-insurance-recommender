from langchain_core.documents import Document

chunk = Document(
    page_content=('27조 (특별약관의 재가입에 관한 사항) 제1항 및 제2항에 따라 재가입하는 경우 또는\n'
 '4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관 제27조 (특별약관의 재\n'
 '가입에 관한 사항) 제5항에 따라 보험계약이 연장된 경우에는 보장개시일(책임개시일)\n'
 '은 이 특별약관의 보험계약일로 봅니다.# 제2조 (보험금 지급에 관한 세부규정)보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 '
 '지급사유에 대해 합의하지 못\n'
 '할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000552',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
