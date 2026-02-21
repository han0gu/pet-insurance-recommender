from langchain_core.documents import Document

chunk = Document(
    page_content=('확인하지 못한 경우(계약자와 연락두절 등으로 회사 안내가\n'
 '계약자에게 도달하지 못한 경우 포함)에는 갱신일 현재의\n'
 '약관 등으로 갱신됩니다. 다만, 계약자는 갱신일 현재의 약\n'
 '관 등에 대해 90일 이내에 그 계약을 취소할 수 있습니다.제4조(갱신보장계약 제1회 보험료의 납입연체와 계약의 해\n'
 '제)\n'
 '\uf000 계약자가 갱신전 보장계약의 보험료를 정상적으로 납입\n'
 '하고, 갱신보장계약의 제1회 보험료를 갱신일까지 납입하지\n'
 '않아 보험료 납입이 연체 중인 경우에 회사는 14일(보험기\n'
 '간이 1년 미만인 경우에는 7일) 이상의 기간을 납입최고(독'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000539',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
