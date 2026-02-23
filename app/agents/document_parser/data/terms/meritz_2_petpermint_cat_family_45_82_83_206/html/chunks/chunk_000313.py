from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자나 피보험자는 청약할 때에 회사<br>가 청약서에서 질문한 중요한 사항에 대해 사실대로 알<br>려야 하며, 위반하는 경우 '
 "계약의 해지 또는 보험금 부<br>지급 등 불이익을 당할 수 있습니다.</p><p id='49' "
 "data-category='paragraph' style='font-size:20px'>【상법 제651조(고지의무위반으로 인한 "
 "계약해지)】</p><br><p id='50' data-category='paragraph' "
 "style='font-size:20px'>보험계약당시에 보험계약자 또는 피보험자가 고의"),
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
