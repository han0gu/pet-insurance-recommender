from langchain_core.documents import Document

chunk = Document(
    page_content=("id='101' data-category='paragraph' style='font-size:14px'>\uf000 전환대상계약에 이 "
 '특별약관이 부가된 이후 제4조(전환 취소)에 따라 전환을 취소<br>한 경우 또는 전환대상계약이 제1조(적용범위) 제1항 제2호에서 정한 '
 '조건을 만족<br>하지 않아 이 특별약관의 효력이 없어진 경우 해당 전환대상계약에는 이 특별약관<br>을 다시 부가할 수 없습니다'),
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
 'indexing': {'chunk_id': 'chunk_001434',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
