from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이하 같다)을 대체공휴일로 한다.<br>상</p><br><p id='133' data-category='paragraph' "
 "style='font-size:16px'>1.</p><br><p id='134' data-category='paragraph' "
 "style='font-size:16px'>2.</p><br><p id='135' data-category='paragraph' "
 "style='font-size:16px'>3.</p><br><p id='136' data-category='list'"),
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
 'indexing': {'chunk_id': 'chunk_000776',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
