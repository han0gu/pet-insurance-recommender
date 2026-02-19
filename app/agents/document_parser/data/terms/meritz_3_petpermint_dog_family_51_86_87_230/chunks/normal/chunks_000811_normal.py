from langchain_core.documents import Document

chunk = Document(
    page_content=('. 라) 뇌졸중, 뇌손상, 척수 및 신경계의 질환 등은 발 병 또는 외상 후 12개월 동안 지속적으로 치료한 후에 장해를 평가한다. '
 '그러나, 12개월이 지났다고 하더라도 뚜렷하게 기능 향상이 진행되고 있는 경우 또는 단기간내 에 사망이 예상되는 경우는 6개월의 범위에서 '
 '장 해 평가를 유보한다. 마) 장해진단 전문의는 재활의학과, 신경외과 또는 신경과 전문의로 한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 226},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible',
            'risk_domains': ['head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000811',
              'chunk_char_len': 205,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
