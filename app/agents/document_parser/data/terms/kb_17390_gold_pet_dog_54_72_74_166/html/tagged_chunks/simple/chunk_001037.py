from langchain_core.documents import Document

chunk = Document(
    page_content=("소염진통제 등)을 사용한 치료 및 항암약물치료 이</p><br><h1 id='8' style='font-size:14px'>외의 방법으로 "
 "시행된 항암치료(항암 방사선치료, 면역치료, 줄기세포치료 등)는</h1><br><h1 id='9' "
 "style='font-size:14px'>제외됩니다.</h1><br><h1 id='10' "
 "style='font-size:14px'>제4조(보험금을 지급하지 않는 사유)</h1><br><p id='11' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001037',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
