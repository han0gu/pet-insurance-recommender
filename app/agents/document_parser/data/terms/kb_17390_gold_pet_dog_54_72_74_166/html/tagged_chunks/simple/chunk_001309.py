from langchain_core.documents import Document

chunk = Document(
    page_content=("id='155' data-category='paragraph' style='font-size:14px'>별</p><br><p "
 "id='156' data-category='paragraph' style='font-size:14px'>약</p><br><p "
 "id='157' data-category='paragraph' style='font-size:14px'>관</p><p id='158' "
 "data-category='paragraph' style='font-size:14px'>상</p><br><p id='159'"),
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
 'indexing': {'chunk_id': 'chunk_001309',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
