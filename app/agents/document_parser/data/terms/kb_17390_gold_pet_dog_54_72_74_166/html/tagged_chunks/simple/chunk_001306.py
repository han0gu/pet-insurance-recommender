from langchain_core.documents import Document

chunk = Document(
    page_content=('id=\'150\' data-category=\'paragraph\' style=\'font-size:16px\'>형) 특별약관"에 의해 '
 "계약자의 선택에 따라 자동갱신으로 운영합니다.</p><br><p id='151' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제1항에 의해 자동갱신을 적용할 경우 보험증권에 그 내용을 기재하여 "
 "드립니다.</p><p id='152' data-category='paragraph' "
 "style='font-size:16px'>제9조(준용규정)<br>이 특별약관에서"),
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
 'indexing': {'chunk_id': 'chunk_001306',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
