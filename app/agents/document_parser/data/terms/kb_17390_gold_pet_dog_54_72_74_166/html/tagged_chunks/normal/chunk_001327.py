from langchain_core.documents import Document

chunk = Document(
    page_content=('. 조향장치의 조작방식, 동력전달방식 또는 원동기 냉각방식 등이 이륜의 자동차<br>와 유사한 구조로 되어 있는 삼륜 또는 사륜의 '
 "자동차로서 승용자동차에 해당<br>하지 않는 자동차</p><br><p id='191' data-category='list' "
 "style='font-size:14px'>3"),
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
 'indexing': {'chunk_id': 'chunk_001327',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
