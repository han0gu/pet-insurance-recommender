from langchain_core.documents import Document

chunk = Document(
    page_content=('. "자기공명영상(MRI)"이란 수의사의 관리 하에 자기공명영상(MRI)을 사용하는<br>도</p><br><p id=\'235\' '
 "data-category='list' style='font-size:14px'>촬영 의료행위를 말합니다.<br>성<br>2. "
 '"컴퓨터단층촬영(CT)"이란 수의사의 관리 하에 자기공명영상(MRI)을 사용하 특<br>는 촬영 의료행위를 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_001025',
              'chunk_char_len': 205,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
