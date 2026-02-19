from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조 (아나필락시스의 정의 및 진단확정)\n'
 '① 이 특별약관에서 「아나필락시스」 이라 함은 한국표준질병·사인분류에 있어서 [별표- 상해관련4]아나필락시스 분류표에서 정한 상병을 '
 '말합니다. ② 「아나필락시스」 의 진단확정은 의료법 제3조(의료기관)에서 규정한 국내의 병원, 의 원 또는 국외의 의료 관련법에 정한 '
 '의료기관의 의사(한의사, 치과의사는 제외합니다) 의 면허를 가진 자에 의하여 내려져야 하며, 이 진단은 임상적 특징 또는 혈액, 항체, '
 '항원검사, 유발검사 및 피부시험 등을 기초로 내려져야 합니다.\n'
 '제4조 (응급실의 정의)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 71},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000370',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
