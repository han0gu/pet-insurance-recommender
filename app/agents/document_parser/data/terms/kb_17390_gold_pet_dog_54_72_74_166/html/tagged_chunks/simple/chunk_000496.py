from langchain_core.documents import Document

chunk = Document(
    page_content=(". 동<br>물</p><br><p id='217' data-category='paragraph' "
 'style=\'font-size:16px\'>경우에는 이 특별약관 계약도 소멸되며 회사는 "보험료 및</p><h1 id=\'218\' '
 "style='font-size:16px'>제4조(준용규정)</h1><br><p id='219' "
 "data-category='paragraph' style='font-size:16px'>이 특별약관에서</p><br><p id='220' "
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
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000496',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
