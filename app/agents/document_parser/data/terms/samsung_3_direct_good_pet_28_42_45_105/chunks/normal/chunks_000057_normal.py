from langchain_core.documents import Document

chunk = Document(
    page_content=('반하고 그 의무가 중요한 사항에 해당하는 경우\n'
 '2. 뚜렷한 위험의 증가와 관련된 제14조(상해보험계약 후 알릴 의무) 제1항에서 정한 계약 후 알릴 의무를 계약자 또는 피보험자의 고의 '
 '또는 중대한 과실로 이행하지 않았을 때\n'
 '② 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 계약을 해지할 수 없습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 33},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000057',
              'chunk_char_len': 184,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
