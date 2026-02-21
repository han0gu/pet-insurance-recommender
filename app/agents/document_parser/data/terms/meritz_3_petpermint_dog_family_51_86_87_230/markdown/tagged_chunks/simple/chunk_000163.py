from langchain_core.documents import Document

chunk = Document(
    page_content=('며 분쟁조정 신청 대상기관은 금융감독원의 금융분쟁조\n'
 '정위원회를 말합니다.\uf000 제2항에 따라 추가적인 조사가 이루어지는 경우, 회사는\n'
 '피보험자의 청구에 따라 회사가 추정하는 보험금의 50% 상\n'
 '당액을 가지급보험금으로 지급합니다.# 【가지급보험금】보험금이 지급기한 내에 지급되지 못할 것으로 판단되는\n'
 '경우 회사가 예상되는 보험금의 일부를 먼저 지급하는\n'
 '제도로 피보험자가 필요로 하는 비용을 보전해 주기 위'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000163',
              'chunk_char_len': 223,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
