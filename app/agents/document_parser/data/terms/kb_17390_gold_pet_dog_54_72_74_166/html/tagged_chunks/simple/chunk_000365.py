from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:18px'>2.</h1><br><p id='26' data-category='paragraph' "
 "style='font-size:14px'>제1조(보험금의 지급사유)</p><br><p id='27' "
 "data-category='paragraph' style='font-size:14px'>회사는 피보험자가 이 특별약관의 보험기간 중에 "
 '상해로 장해분류표(【별표1】(장해<br>분류표) 참조'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000365',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
