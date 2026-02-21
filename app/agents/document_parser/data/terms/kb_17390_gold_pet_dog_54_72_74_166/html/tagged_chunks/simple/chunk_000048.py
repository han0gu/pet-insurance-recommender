from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 지급예정일은 다음 각 호의 어느 하나에<br>해당하는 경우를 제외하고는 제7조(보험금의 청구)에서 정한 서류를 접수한 '
 "날부<br>터 30영업일 이내에서 정합니다.<br>1. 소송제기</p><br><p id='60' "
 "data-category='list'></p><br><p id='61' data-category='list' "
 "style='font-size:14px'>2. 분쟁조정 신청<br>3. 수사기관의 조사<br>4"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000048',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
