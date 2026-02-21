from langchain_core.documents import Document

chunk = Document(
    page_content=('전 알릴 의무)<br>계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 건<br>강진단할 때를 말합니다) 청약서에서 질문한 사항에 '
 '대하여<br>알고 있는 사실을 반드시 사실대로 알려야(이하「계약 전<br>알릴 의무」라 하며, 상법상「고지의무」와 같습니다) '
 '합니<br>다'),
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
