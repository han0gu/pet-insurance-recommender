from langchain_core.documents import Document

chunk = Document(
    page_content=('- 되지 않을 수 있습니다.다만, 회사는 계약자 등이 분쟁조정을 신청했다는 사유만으로 이자지\n'
 '- 급을 거절하지 않습니다.\n'
 '- 4. 가산이율 적용시 「보험금의 지급절차」 제2항 각 호의 어느 하나에 해당되는 사유로 지연된\n'
 '- 경우에는 해당기간에 대하여 가산이율을 적용하지 않습니다.(다만, 상해 · 질병 관련 보장에\n'
 '- 한합니다)\n'
 '- 5. 가산이율 적용시 금융위원회 또는 금융감독원이 정당한 사유로 인정하는 경우에는 해당 기간\n'
 '- 에 대하여 가산이율을 적용하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000726',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
