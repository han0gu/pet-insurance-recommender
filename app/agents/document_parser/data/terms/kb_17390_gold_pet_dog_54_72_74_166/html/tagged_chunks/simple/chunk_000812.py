from langchain_core.documents import Document

chunk = Document(
    page_content=('. 약<br>관<br>\uf000 회사는 제5항의 서면조사에 대한 동의 요청시 조사목적, 사용처 등을 명시하고</p><br><p '
 "id='179' data-category='paragraph' style='font-size:16px'>설명합니다.</p><br><p "
 "id='180' data-category='paragraph' style='font-size:14px'>별</p><p id='181' "
 "data-category='paragraph' style='font-size:14px'>제6조(보험금의 분담)<br>\uf000 회사는 "
 '이 계약에서 보장하는'),
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
 'indexing': {'chunk_id': 'chunk_000812',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
