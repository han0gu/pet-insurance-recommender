from langchain_core.documents import Document

chunk = Document(
    page_content=("id='193' data-category='paragraph' style='font-size:20px'>4.</p><br><p "
 "id='194' data-category='paragraph' "
 "style='font-size:20px'>천식지속상태(급성중증천식)진단비</p><br><p id='195' "
 "data-category='paragraph' style='font-size:14px'>및</p><p id='196' "
 "data-category='paragraph' style='font-size:16px'>제1조(보험금의"),
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
 'indexing': {'chunk_id': 'chunk_000652',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
