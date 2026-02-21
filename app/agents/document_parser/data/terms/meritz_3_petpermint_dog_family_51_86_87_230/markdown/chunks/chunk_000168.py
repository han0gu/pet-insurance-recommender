from langchain_core.documents import Document

chunk = Document(
    page_content=('경우에도 회사의 제1항에 따른 지급보험금 결정에는 영향을\n'
 '미치지 않습니다.# 제7조(계약 전 알릴 의무)계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 건\n'
 '강진단할 때를 말합니다) 청약서에서 질문한 사항에 대하여\n'
 '알고 있는 사실을 반드시 사실대로 알려야(이하 「계약 전\n'
 '알릴 의무」라 하며, 상법상「고지의무」와 같습니다) 합니\n'
 '다.94# 【계약 전 알릴 의무】상법 제651조(고지의무위반으로 인한 계약해지)에서 정\n'
 '하고 있는 의무. 계약자나 피보험자는 청약할 때에 회사\n'
 '가 청약서에서 질문한 중요한 사항에 대해 사실대로 알'),
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
