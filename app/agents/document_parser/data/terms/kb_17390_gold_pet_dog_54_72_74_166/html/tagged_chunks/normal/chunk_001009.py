from langchain_core.documents import Document

chunk = Document(
    page_content=('30일" data-coord="top-left:(147,115); bottom-right:(728,224)" '
 "/></figure><br><h1 id='216' style='font-size:16px'>2024년 4월 10일</h1><br><p "
 "id='217' data-category='paragraph' style='font-size:16px'>2024년 5월 "
 "9일</p><br><h1 id='218' style='font-size:16px'>- 단, 상해(상해로 인한 창상 또는 교상, 이물섭취를 "
 '포함)를 직접적인'),
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
 'indexing': {'chunk_id': 'chunk_001009',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
