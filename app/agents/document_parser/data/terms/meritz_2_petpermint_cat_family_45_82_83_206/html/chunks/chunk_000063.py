from langchain_core.documents import Document

chunk = Document(
    page_content=("변경)</h1><br><p id='86' data-category='paragraph' "
 "style='font-size:20px'>\uf000 계약자(보험금 지급사유 발생 후에는 보험수익자)는 회<br>사의 사업방법서에서 정한 "
 "바에 따라 보험금의 전부 또는</p><footer id='87' style='font-size:14px'>55</footer><p "
 "id='88' data-category='paragraph' style='font-size:16px'>일부에 대하여 나누어 지급받거나 "
 '일시에 지급받는 방법으<br>로 변경할 수'),
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
