from langchain_core.documents import Document

chunk = Document(
    page_content=('아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보\n'
 '된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)\n'
 '4. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류② 제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 '
 '병원이나 의\n'
 '원 또는 국외의 의료관련법에서 정한 의료기관에서 발급한 것이어야 합니다.# 제7조 (특별약관의 소멸)피보험자가 보험기간 중에 사망하였을 '
 '경우에는 "보험료 및 해약환급금 산출방법서"에서'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000318',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
