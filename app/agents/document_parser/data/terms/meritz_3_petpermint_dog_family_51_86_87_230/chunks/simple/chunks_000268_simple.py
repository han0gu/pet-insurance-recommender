from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항에 따라 해지계약을 부활(효력회복)하는 경우에는 제7조(계약 전 알릴 의무), 제9조(알릴 의무 위반의 효과), '
 '제10조(사기에 의한 계약), 제11조(보험계약의 성립) 및 제 16조(제1회 보험료 및 회사의 보장개시)의 규정을 준용합니 다. 이 때 '
 '회사는 해지 전 발생한 보험금 지급사유를 이유 로 부활(효력회복)을 거절하지 않습니다. \uf000 제1항에서 정한 계약의 부활이 '
 '이루어진 경우라도 계약'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 105},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000268',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
