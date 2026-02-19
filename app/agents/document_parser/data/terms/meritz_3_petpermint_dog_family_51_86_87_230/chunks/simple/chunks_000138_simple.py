from langchain_core.documents import Document

chunk = Document(
    page_content=('【 부활(효력회복) 】\n'
 '보험료 납입을 연체하여 계약이 해지되고 계약자가 해약 환급금을 받지 않은 경우 회사가 정하는 소정의 절차에 따라 해지된 계약을 다시 '
 '되살리는 것을 말합니다.\n'
 '\uf000 제1항에 따라 해지계약을 부활(효력회복)하는 경우에는 제15조(계약 전 알릴 의무), 제17조(알릴 의무 위반의 효 과), '
 '제18조(사기에 의한 계약), 제19조(보험계약의 성립) 및 제26조(제1회 보험료 및 회사의 보장개시)의 규정을 준 용합니다. 이 때 '
 '회사는 해지 전 발생한 보험금 지급사유를 이유로 부활(효력회복)을 거절하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 78},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000138',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
