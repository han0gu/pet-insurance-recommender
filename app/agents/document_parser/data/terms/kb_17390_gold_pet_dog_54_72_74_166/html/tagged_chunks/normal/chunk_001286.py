from langchain_core.documents import Document

chunk = Document(
    page_content=('않는 사유) 및 다음<br>중 어느 한 가지의 경우로 인하여 보험금 지급사유가 발생한 때에는 보험금을 지<br>급하지 '
 "않습니다.</p><br><p id='123' data-category='list' style='font-size:14px'>1"),
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
 'indexing': {'chunk_id': 'chunk_001286',
              'chunk_char_len': 136,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
