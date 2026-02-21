from langchain_core.documents import Document

chunk = Document(
    page_content=('- 대한 과실로 반려동물 비용손해 관련 특별약관 일반조\n'
 '- 항 제7조(계약 전 알릴 의무)를 위반하고 그 의무가\n'
 '- 중요한 사항에 해당하는 경우\n'
 '- ② 뚜렷한 위험의 증가와 관련된 제12조(계약 후 알릴 의\n'
 '- 무) 제1항에서 정한 계약 후 알릴 의무를 계약자 또는\n'
 '- 피보험자의 고의 또는 중대한 과실로 이행하지 않았을\n'
 '- 때\n'
 '- ③ 상당한 이유없이 손해조사를 거부 또는 회피할 때\n'
 '\uf000 제1항 제1호에도 불구하고 다음 중 한가지의 경우에 해'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000508',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
