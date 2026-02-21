from langchain_core.documents import Document

chunk = Document(
    page_content=('비루관 관련 질환으로 인한 비용\n'
 '특# \uf000 제2항에서 정한 조치에 다른 진료를 병행하여 실시한 경우에는 제2항에서 정한 조치(마취 비용을 포함합니다.)에대한 '
 '보험금은 지급하지 않습니다.관\n'
 '제4조(수술의 정의와 장소)\n'
 '이 특별약관에 있어서 "수술"이라 함은 동물병원의 수의사 자격을 가진 자(이하 "수\n'
 '의사"라 합니다)에 의하여 치료가 필요하다고 인정된 상해 또는 질병 치료를 위하여\n'
 '수의사법 제17조(개설)에서 규정한 국내의 동물병원에서 수의사의 관리 하에 직접적'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000566',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
