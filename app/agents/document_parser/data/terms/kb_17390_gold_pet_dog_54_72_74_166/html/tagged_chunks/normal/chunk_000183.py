from langchain_core.documents import Document

chunk = Document(
    page_content=("63</p><br><p id='234' data-category='paragraph' style='font-size:18px'>- 63 "
 "-</p><br><p id='235' data-category='paragraph' "
 "style='font-size:20px'>사항</p><p id='236' data-category='paragraph' "
 "style='font-size:14px'>관</p><table id='0'"),
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
 'indexing': {'chunk_id': 'chunk_000183',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
