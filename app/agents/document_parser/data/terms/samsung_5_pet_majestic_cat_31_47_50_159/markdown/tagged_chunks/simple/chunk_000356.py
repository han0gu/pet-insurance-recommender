from langchain_core.documents import Document

chunk = Document(
    page_content=('- 로 지급합니다. 다만, 동일부위에 대한 성형수술을 2회 이상 받은 경우에는 최초로 받\n'
 '- 은 수술에 대해서만 지급합니다.\n'
 '<용어풀이>[안면부, 상지, 하지]- 1. 안면부란 이마를 포함하여 목까지의 얼굴부분을 말합니다.\n'
 '- 2. 상지란 어깨관절 이하의 팔과 손가락 부분을 말합니다.\n'
 '# 3. 하지란 엉덩이관절 이하 다리와 발가락 부분을 말합니다.# 제2조 (보험금 지급에 관한 세부규정)보험수익자와 회사가 '
 '제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000356',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
