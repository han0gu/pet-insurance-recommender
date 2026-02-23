from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 건\n'
 '강진단할 때를 말합니다) 청약서에서 질문한 사항에 대하여\n'
 '알고 있는 사실을 반드시 사실대로 알려야(이하「계약 전\n'
 '알릴 의무」라 하며, 상법상「고지의무」와 같습니다) 합니\n'
 '다. 다만, 진단계약의 경우 의료법 제3조(의료기관)의 규정\n'
 '에 따른 종합병원과 병원에서 직장 또는 개인이 실시한 건\n'
 '강진단서 사본 등 건강상태를 판단할 수 있는 자료로 건강\n'
 '진단을 대신할 수 있습니다.57# 【 계약 전 알릴 의무 】상법 제651조(고지의무위반으로 인한 계약해지)에서 정'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
