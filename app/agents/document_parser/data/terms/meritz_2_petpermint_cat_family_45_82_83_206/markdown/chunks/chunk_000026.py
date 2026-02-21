from langchain_core.documents import Document

chunk = Document(
    page_content=('정과 관련하여 확정된 장해지급률에 따른 보험금을 초과한\n'
 '부분에 대한 분쟁으로 보험금 지급이 늦어지는 경우에는 보\n'
 '험수익자의 청구에 따라 이미 확정된 보험금을 먼저 가지급\n'
 '합니다.\n'
 '\uf000 제2항에 따라 추가적인 조사가 이루어지는 경우, 회사는\n'
 '보험수익자의 청구에 따라 회사가 추정하는 보험금의 50%\n'
 '상당액을 가지급보험금으로 지급합니다.# 【가지급보험금】보험금이 지급기한 내에 지급되지 못할 것으로 판단되는 경우 회\n'
 '사가 예상되는 보험금의 일부를 먼저 지급하는 제도로 피보험자\n'
 '가 필요로 하는 비용을 보전해 주기 위해 회사가 먼저 지급하는'),
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
