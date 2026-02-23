from langchain_core.documents import Document

chunk = Document(
    page_content=('2024년 4월 10일 2024년 5월 9일" data-coord="top-left:(150,630); '
 'bottom-right:(738,735)" /></figure><br><p id=\'141\' '
 "data-category='paragraph' style='font-size:14px'>- 단, 상해를 직접적인 원인으로 치료를 받은 "
 "경우에는 보장개시일은 보험계약</p><br><p id='142' data-category='paragraph' "
 "style='font-size:14px'>일로 합니다.</p><br><h1 id='143'"),
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
 'indexing': {'chunk_id': 'chunk_000953',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
