from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>반</p><p id='284' data-category='paragraph' "
 "style='font-size:18px'>- 83 -</p><br><p id='285' data-category='paragraph' "
 "style='font-size:14px'>질</p><br><p id='286' data-category='paragraph' "
 "style='font-size:14px'>병</p><br><p id='287' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000536',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
