from langchain_core.documents import Document

chunk = Document(
    page_content=('같은 상해를 직접적인 원인으<br>로 2가지 이상의 치아파절 발생시에는 1회에 한하여 치아파절진단비를 '
 '지급합<br>니다.<br>\uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합<br>의하지 못할 때는 '
 '보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견<br>에 따를 수 있습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000504',
              'chunk_char_len': 175,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
