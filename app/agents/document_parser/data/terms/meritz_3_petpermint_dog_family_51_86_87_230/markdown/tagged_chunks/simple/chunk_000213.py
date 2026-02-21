from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 회사가 부활(효력회복)을 승낙한 때에 계약자는 부활\n'
 '(효력회복)을 청약한 날까지의 연체된 보험료와 이에 대한\n'
 '연체된 이자(보장보험료에 대해서 평균공시이율+1%로 계산\n'
 '한 이자)를 더하여 납입하여야 합니다.# 【 부활(효력회복) 】보험료 납입을 연체하여 계약이 해지되고 계약자가 해약\n'
 '환급금을 받지 않은 경우 회사가 정하는 소정의 절차에\n'
 '따라 해지된 계약을 다시 되살리는 것을 말합니다.\uf000 제1항에 따라 해지계약을 부활(효력회복)하는 경우에는\n'
 '제7조(계약 전 알릴 의무), 제9조(알릴 의무 위반의 효과),'),
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
 'indexing': {'chunk_id': 'chunk_000213',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
