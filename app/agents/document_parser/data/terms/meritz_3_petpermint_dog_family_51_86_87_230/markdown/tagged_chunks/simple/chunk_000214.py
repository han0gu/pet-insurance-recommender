from langchain_core.documents import Document

chunk = Document(
    page_content=('제7조(계약 전 알릴 의무), 제9조(알릴 의무 위반의 효과),\n'
 '제10조(사기에 의한 계약), 제11조(보험계약의 성립) 및 제\n'
 '16조(제1회 보험료 및 회사의 보장개시)의 규정을 준용합니\n'
 '다. 이 때 회사는 해지 전 발생한 보험금 지급사유를 이유\n'
 '로 부활(효력회복)을 거절하지 않습니다.\n'
 '\uf000 제1항에서 정한 계약의 부활이 이루어진 경우라도 계약105자 또는 피보험자가 최초계약 청약시(2회 이상 부활이 이루\n'
 '어진 경우 종전 모든 부활 청약 포함) 제7조(계약 전 알릴\n'
 '의무)를 위반한 경우에는 제9조(알릴 의무 위반의 효과)가'),
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
 'indexing': {'chunk_id': 'chunk_000214',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
