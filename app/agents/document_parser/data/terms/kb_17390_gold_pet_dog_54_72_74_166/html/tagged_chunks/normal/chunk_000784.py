from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험증권에 기재된 피보험자(이하 "피보험자 본인"이라 합니다) 성<br>특</p><br><p id=\'145\' '
 "data-category='paragraph' style='font-size:16px'>함은 아래에 정한 보험증권에 기재된 피보험자 및 "
 "그</p><br><p id='146' data-category='paragraph' "
 "style='font-size:14px'>해</p><br><p id='147' data-category='paragraph' "
 "style='font-size:14px'>상</p><p id='148'"),
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
 'indexing': {'chunk_id': 'chunk_000784',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
