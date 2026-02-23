from langchain_core.documents import Document

chunk = Document(
    page_content=('의하여 장해지급률의 판정 및 지급할 보험금의 결정과 관련하여 확정된 장<br>해지급률에 따른 보험금을 초과한 부분에 대한 분쟁으로 보험금 '
 '지급이 늦어지는<br>경우에는 보험수익자의 청구에 따라 이미 확정된 보험금을 먼저 가지급합니다.<br>\uf000 제2항에 의하여 '
 "추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에 따</p><br><table id='67' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>라 회사가 "
 '추정하는</td><td></td><td></td><td>보험금의'),
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
 'indexing': {'chunk_id': 'chunk_000052',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
