from langchain_core.documents import Document

chunk = Document(
    page_content=("사유)</h1><br><p id='11' data-category='paragraph' "
 "style='font-size:14px'>\uf000</p><br><p id='12' data-category='list' "
 "style='font-size:14px'>회사는 아래의 사유로 인한 손해는 보상하지 않습니다.<br>1"),
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
 'indexing': {'chunk_id': 'chunk_001038',
              'chunk_char_len': 170,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
