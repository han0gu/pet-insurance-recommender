from langchain_core.documents import Document

chunk = Document(
    page_content=('생 후에 이루어진 경우에는 제16조(상해보험계약 후 알릴\n'
 '의무) 제4항 또는 제5항에 따라 보험금을 지급합니다.\n'
 '\uf000 제1항에도 불구하고 알릴 의무를 위반한 사실이 보험금\n'
 '지급사유 발생에 영향을 미쳤음을 회사가 증명하지 못한 경\n'
 '우에는 제4항 및 제5항에 관계없이 약정한 보험금을 지급합\n'
 '니다.\n'
 '\uf000 회사는 다른 보험가입내역에 대한 계약 전 알릴 의무 위\n'
 '반을 이유로 계약을 해지하거나 보험금 지급을 거절하지 않\n'
 '습니다.\n'
 '\uf000 제30조(보험료의 납입을 연체하여 해지된 계약의 부활\n'
 '(효력회복))에 따라 이 계약이 부활이 이루어진 경우에는'),
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
 'indexing': {'chunk_id': 'chunk_000060',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
