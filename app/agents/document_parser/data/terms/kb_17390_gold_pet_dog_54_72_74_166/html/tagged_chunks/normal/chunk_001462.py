from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>해</p><br><p id='143' data-category='paragraph' "
 "style='font-size:14px'>및</p><br><p id='144' data-category='paragraph' "
 "style='font-size:14px'>질</p><br><p id='145' data-category='paragraph' "
 "style='font-size:14px'>병</p><p id='146' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001462',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
