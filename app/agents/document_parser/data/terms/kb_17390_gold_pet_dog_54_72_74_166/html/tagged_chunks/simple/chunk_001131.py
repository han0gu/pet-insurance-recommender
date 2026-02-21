from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 부활<br>(효력회복)일을 보험계약일로 하여 제1조(보험금의 지급사유) 제6항을 '
 "적용합니다.<br>제<br>도</p><br><h1 id='140' "
 "style='font-size:16px'>제9조(준용규정)</h1><br><p id='141' "
 "data-category='paragraph' style='font-size:16px'>제17조(보험료</p><br><p id='142' "
 "data-category='paragraph' style='font-size:14px'>물</p><h1 id='143'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001131',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
