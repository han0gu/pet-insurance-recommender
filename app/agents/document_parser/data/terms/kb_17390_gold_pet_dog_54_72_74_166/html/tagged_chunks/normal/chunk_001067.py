from langchain_core.documents import Document

chunk = Document(
    page_content=("id='40' data-category='paragraph' style='font-size:14px'>이 특별약관에 있어서 "
 '"수술"이라 함은 동물병원의 수의사 자격을 가진 자(이하 "수<br>의사"라 합니다)에 의하여 치료가 필요하다고 인정된 상해 또는 질병 '
 '치료를 위하여<br>수의사법 제17조(개설)에서 규정한 국내의 동물병원에서 수의사의 관리 하에 직접적<br>인 치료를 목적으로 기구를 '
 '사용하여 생체에 절단, 절제 등의 조작을 가하는 것을 말<br>합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001067',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
