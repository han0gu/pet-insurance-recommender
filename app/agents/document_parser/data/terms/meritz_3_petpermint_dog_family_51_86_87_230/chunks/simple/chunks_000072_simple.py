from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 제1항 제2호에 따른 계약의 해지가 보험금 지급사유 발 생 후에 이루어진 경우에는 제16조(상해보험계약 후 알릴 의무) '
 '제4항 또는 제5항에 따라 보험금을 지급합니다. \uf000 제1항에도 불구하고 알릴 의무를 위반한 사실이 보험금 지급사유 발생에 영향을 '
 '미쳤음을 회사가 증명하지 못한 경 우에는 제4항 및 제5항에 관계없이 약정한 보험금을 지급합 니다. \uf000 회사는 다른 '
 '보험가입내역에 대한 계약 전 알릴 의무 위 반을 이유로 계약을 해지하거나 보험금 지급을 거절하지 않 습니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 66},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000072',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
