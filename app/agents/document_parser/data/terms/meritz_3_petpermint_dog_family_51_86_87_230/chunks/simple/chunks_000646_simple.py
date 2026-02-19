from langchain_core.documents import Document

chunk = Document(
    page_content=('① 갱신될 자동갱신 적용대상 계약(이하「갱신보장계약」 이라 합니다)이 끝나는 날이 회사가 정한 기간 내일 것 ② 갱신일에 있어서 '
 '피보험자의 나이가 회사가 정한 나이 의 범위 내일 것 ③ 보통약관 제29조(보험료의 납입이 연체되는 경우 납입 최고(독촉)와 계약의 '
 '해지)에서 정한 납입최고(독촉) 기간 내에 갱신전 보장계약의 보험료가 납입완료 되었 을 것'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 189},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000646',
              'chunk_char_len': 196,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
