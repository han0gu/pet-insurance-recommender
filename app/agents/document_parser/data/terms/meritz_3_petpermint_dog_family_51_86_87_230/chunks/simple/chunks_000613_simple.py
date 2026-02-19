from langchain_core.documents import Document

chunk = Document(
    page_content=('여부에 관계없이 그 사실을 안 날부터 1개월 이내에 이 계 약을 해지할 수 있습니다.\n'
 '① 계약자, 피보험자 또는 이들의 대리인이 고의 또는 중 대한 과실로 반려동물 비용손해 관련 특별약관 일반조 항 제7조(계약 전 알릴 '
 '의무)를 위반하고 그 의무가 중요한 사항에 해당하는 경우 ② 뚜렷한 위험의 증가와 관련된 제12조(계약 후 알릴 의 무) 제1항에서 정한 '
 '계약 후 알릴 의무를 계약자 또는 피보험자의 고의 또는 중대한 과실로 이행하지 않았을 때 ③ 상당한 이유없이 손해조사를 거부 또는 회피할 '
 '때'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 182},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000613',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
