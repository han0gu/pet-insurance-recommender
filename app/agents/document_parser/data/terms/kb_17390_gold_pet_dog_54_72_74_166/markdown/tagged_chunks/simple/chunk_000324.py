from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보장합니다)\n'
 '- 2. 선천적 기형 및 이에 근거한 병상\n'
 '# 제4조(수술의 정의와 장소)# \uf000 이 특별약관에 있어서 "수술"이라함은 병원 또는 의원의 의사, 치과의사 면허를- 가진 '
 '자(이하 "의사"라 합니다)에 의하여 치료가 필요하다고 인정한 경우로서 자\n'
 '- 택 등에서 치료가 곤란하여 의료기관에서 의사의 관리 하에 직접적인 치료를 목\n'
 '- 적으로 의료기구를 사용하여 생체(生體)에 절단(切斷), 절제(切除) 등의 조작(操\n'
 '- 作)을 가하는 것을 말합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000324',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
