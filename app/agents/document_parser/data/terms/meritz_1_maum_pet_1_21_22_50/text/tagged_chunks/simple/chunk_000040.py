from langchain_core.documents import Document

chunk = Document(
    page_content=('할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있습니\n'
 '다. 제3자는 수의사법 제2조(정의)에 규정한 동물병원 소속의 수의사 중에서 정하며,- 7 -보험금 지급사유 판정에 드는 의료비용은 '
 '회사가 전액 부담합니다.제10조(지급보험금의 계산)① 동일한 반려동물과 동일한 사고에 관하여 보험금을 지급하는 다른 계약(공제계약을 포\n'
 '함합니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한\n'
 '지급보험금의 합계액이 피보험자가 부담한 비용금액을 초과할 때에는 아래에 따라 보'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000040',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
