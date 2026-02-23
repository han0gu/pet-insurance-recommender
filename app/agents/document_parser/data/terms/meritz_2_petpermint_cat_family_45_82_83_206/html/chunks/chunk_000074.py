from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 그 대표자는 각각 다<br>른 계약자 또는 보험수익자를 대리하는 것으로 합니다.<br>\uf000 지정된 계약자 또는 '
 '보험수익자의 소재가 확실하지 않은<br>경우에는 이 계약에 관하여 회사가 계약자 또는 보험수익자<br>1명에 대하여 한 행위는 각각 다른 '
 '계약자 또는 보험수익자<br>에게도 효력이 미칩니다.<br>\uf000 계약자가 2명 이상인 경우에는 그 책임을 연대로 '
 "합니<br>다.</p><br><h1 id='6' style='font-size:18px'>【계약자가 2명 이상인 경우 "
 "】</h1><br><p id='7'"),
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
