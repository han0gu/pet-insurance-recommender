from langchain_core.documents import Document

chunk = Document(
    page_content=("id='12' style='font-size:16px'>관 련 법 규</h1><br><h1 id='13' "
 "style='font-size:16px'>관공서의 공휴일에 관한 규정 제2조 및 제3조</h1><br><p id='14' "
 "data-category='paragraph' style='font-size:18px'>통<br>제2조(공휴일)<br>관공서의 공휴일은 "
 '다음 각 호와 같다'),
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
 'indexing': {'chunk_id': 'chunk_000017',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
