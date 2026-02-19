from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조 (보험금의 분담)\n'
 '① 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합 니다)이 있을 경우 각 계약에 대하여 다른 계약이 '
 '없는 것으로 하여 각각 산출한 보상 책임액의 합계액이 손해액을 초과할 때에는 회사는 아래에 따라 손해를 보상합니다.\n'
 '<지급보험금 계산방법> 다른 계약이 없을 때 이 계약의 지급보험금 피보험자가 부담한 의료비 × 다른 계약이 없는 것으로 하여 각각 계약의 '
 '지급보험금의 합계액'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 116},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000733',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
