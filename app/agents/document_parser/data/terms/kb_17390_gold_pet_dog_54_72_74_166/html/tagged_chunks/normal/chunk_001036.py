from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 항암 치료가 아닌 다른 질환 치료</p><br><p id='6' data-category='paragraph' "
 "style='font-size:14px'>114 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='7' "
 "data-category='paragraph' style='font-size:14px'>를 목적으로 사용되나 항암 효과가 있어 부수적으로 "
 '항암제로 쓰이는 약물(스테로<br>이드제, 면역억제제, 항생제, 소염진통제 등)을 사용한 치료 및 항암약물치료 이</p><br><h1 '
 "id='8'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001036',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
