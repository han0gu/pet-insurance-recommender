from langchain_core.documents import Document

chunk = Document(
    page_content=(". 재가입일에 있어서 반려동물의 나이가 회사가 최초가입 당시 정한 재가입 나이</p><br><p id='101' "
 "data-category='list' style='font-size:14px'>의 범위 내일 것<br>2"),
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
 'indexing': {'chunk_id': 'chunk_000919',
              'chunk_char_len': 121,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
