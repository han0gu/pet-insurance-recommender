from langchain_core.documents import Document

chunk = Document(
    page_content=("제도성 특별약관</p><p id='180' data-category='paragraph' style='font-size:14px'>- "
 "131 -</p><h1 id='181' style='font-size:20px'>제5장 제도성 특별약관</h1><h1 id='182' "
 "style='font-size:18px'>1"),
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
 'indexing': {'chunk_id': 'chunk_001316',
              'chunk_char_len': 174,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
