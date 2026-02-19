from langchain_core.documents import Document

chunk = Document(
    page_content=('제5조 (의무보험과의 관계)\n'
 '① 회사는 이 특별약관에 의하여 보상하여야 하는 금액이 의무보험에서 보상하는 금액을 초과할 때에 한하여 그 초과액만을 보상합니다. 다만, '
 '의무보험이 다수인 경우에는 제 10조(보험금의 분담)를 따릅니다. 제1항의 의무보험은 피보험자가 법률에 의하여 의무적으로 가입하여야 하는 '
 '보험으로 서 공제계약을 포함합니다. 피보험자가 의무보험에 가입하여야 함에도 불구하고 가입하지 않은 경우에는 그가 가 입했더라면 '
 '의무보험에서 보상했을 금액을 제1항의 "의무보험에서 보상하는 금액" 으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000569',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
