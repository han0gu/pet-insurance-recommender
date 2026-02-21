from langchain_core.documents import Document

chunk = Document(
    page_content=('하는 다른 계약(공제계약을 포함합니다)이 있을 경우 각 계\n'
 '약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 지\n'
 '급보험금의 합계액이 피보험자가 부담한 비용금액을 초과할\n'
 '때에는 아래에 따라 보험금을 지급합니다.피보험자가 이 계약의 지급보험금\n'
 '부담한 총 × 다른 계약이 없는 것으로 하여 각각 계산한\n'
 '비용금액 지급보험금의 합계액\uf000 피보험자가 다른 계약에 대하여 보험금 청구를 포기한\n'
 '경우에도 회사의 제1항에 따른 지급보험금 결정에는 영향을'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
