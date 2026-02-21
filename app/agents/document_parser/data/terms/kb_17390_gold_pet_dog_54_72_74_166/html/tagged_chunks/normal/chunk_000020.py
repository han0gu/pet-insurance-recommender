from langchain_core.documents import Document

chunk = Document(
    page_content=(". 6월 6일 (현충일)</p><br><p id='17' data-category='paragraph' "
 "style='font-size:16px'>9.</p><br><p id='18' data-category='paragraph' "
 "style='font-size:16px'>(음력 8월 14일, 15일, 16일)</p><br><p id='19' "
 "data-category='paragraph' style='font-size:16px'>추석 전날, 추석, 추석 다음날</p><br><p "
 "id='20'"),
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
 'indexing': {'chunk_id': 'chunk_000020',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
