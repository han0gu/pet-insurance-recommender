from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항의 의무보험은 피보험자가 법률에 의하여 의무적으 로 가입하여야 하는 보험으로서 공제계약을 포함합니다. \uf000 '
 '피보험자가 의무보험에 가입하여야 함에도 불구하고 가 입하지 않은 경우에는 그가 가입했더라면 의무보험에서 보 상했을 금액을 제1항의 '
 '의무보험에서 보상하는 금액으로 봅 니다.\n'
 '제7조(보험금의 분담)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 178},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000595',
              'chunk_char_len': 172,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
