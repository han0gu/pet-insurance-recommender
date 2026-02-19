from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자 또는 피보험자가 고의 또는 중대한 과실로 제 15조(계약 전 알릴 의무)를 위반하고 그 의무가 중요 한 사항에 해당하는 경우 '
 '② 뚜렷한 위험의 증가와 관련된 제16조(상해보험계약 후 알릴 의무) 제1항에서 정한 계약 후 알릴 의무를 계약 자 또는 피보험자의 고의 '
 '또는 중대한 과실로 이행하 지 않았을 때\n'
 '\uf000 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당 하는 경우에는 회사는 계약을 해지할 수 없습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 65},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000065',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
