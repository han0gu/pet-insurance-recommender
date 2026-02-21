from langchain_core.documents import Document

chunk = Document(
    page_content=('니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 보상\n'
 '책임액의 합계액이 손해액을 초과할 때에는 회사는 아래에 따라 손해를 보상합니다.![image](/image/placeholder)\n'
 '<지급보험금 계산방법>\n'
 '다른 계약이 없을 때 이 계약의 지급보험금\n'
 '피보험자가 부담한 의료비 × 다른 계약이 없는 것으로 하여 각각 계약의 지급보험금의\n'
 '합계액② 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000579',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
