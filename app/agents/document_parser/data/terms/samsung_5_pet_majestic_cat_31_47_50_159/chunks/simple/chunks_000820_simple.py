from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 제1항 제2호에도 불구하고 계약 전 알릴 의무를 위반하고 계약자가 보험계약의 변경 에 대한 청약을 하지 않는 경우 회사는 보통약관 '
 '「계약 전 알릴 의무 위반의 효과」 조항에 따라 보험계약을 해지할 수 있습니다. ④ 이 특별약관에 대한 회사의 보장개시일(책임개시일)은 '
 '보험계약 「제1회 보험료 및 회 사의 보장개시」에서 정한 보장개시일(책임개시일)과 동일합니다. ⑤ 보험계약이 해지, 기타 사유에 의하여 '
 '효력이 없게 된 경우에는 이 특별약관도 더 이 상 효력이 없습니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 129},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000820',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
