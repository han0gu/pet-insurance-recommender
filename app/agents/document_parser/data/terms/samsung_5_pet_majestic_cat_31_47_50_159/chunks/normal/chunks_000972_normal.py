from langchain_core.documents import Document

chunk = Document(
    page_content=('. 라) 뇌졸중, 뇌손상, 척수 및 신경계의 질환 등은 발병 또는 외상 후 12개월 동 안 지속적으로 치료한 후에 장해를 평가한다. '
 '그러나, 12개월이 지났다고 하더라도 뚜렷하게 기능 향상이 진행되고 있는 경우 또는 단기간 내에 사 망이 예상되는 경우는 6개월의 범위 '
 '내에서 장해 평가를 유보한다. 마) 장해진단 전문의는 재활의학과, 신경외과 또는 신경과 전문의로 한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 147},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000972',
              'chunk_char_len': 207,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
