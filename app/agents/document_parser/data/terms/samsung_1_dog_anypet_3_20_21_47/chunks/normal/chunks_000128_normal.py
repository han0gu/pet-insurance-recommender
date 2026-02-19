from langchain_core.documents import Document

chunk = Document(
    page_content=("【수술】 동물병원의 수의사 자격을 가진 자(이하 '수의사'라 합니다)에 의하여 치료가 필요하다고 인정된 상해 또는 질병 치료를 위하여 "
 '수의사법 제17조(개설)에서 규정한 국내의 동물병원에서 수의사의 관리 하에 직접적 인 치료를 목적으로 기구를 사용하여 생체에 절개, '
 '절단, 절제 등의 조작을 가하는 것을 말합니다. 단 수술에 서 아래에 정한 사항은 제외합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 24},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000128',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
