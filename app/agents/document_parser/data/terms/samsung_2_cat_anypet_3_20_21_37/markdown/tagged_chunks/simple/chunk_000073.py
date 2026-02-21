from langchain_core.documents import Document

chunk = Document(
    page_content=('습니다.1. 계약자 또는 피보험자가 보험금을 지급받을 목적으로 고의로 보험금 지급사유를 발생시킨 경우당신에게 좋은보험 삼성화재- 16 '
 '-2. 계약자 또는 피보험자가 보험금 청구에 관한 서류에 고의로 사실과 다른 것을 기재하였거나 그\n'
 '서류 또는 증거를 위조 또는 변조한 경우. 다만, 이미 보험금 지급사유가 발생한 경우에는 보험\n'
 '금 지급에 영향을 미치지 않습니다.【설명】 계약자, 피보험자 또는 보험수익자가 보험금 청구에 관한 서류에 고의로 사실과 다른 것을 기'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000073',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
