from langchain_core.documents import Document

chunk = Document(
    page_content=('장해상태가 되었을 때에는 보험수익자에게 최초 1<br>회에 한하여 이 보장의 보험가입금액 전액을 일반상해80%이<br>상후유장해보험금으로 '
 "지급합니다.</p><p id='32' data-category='paragraph' "
 "style='font-size:18px'>제4조(보험금 지급에 관한 세부규정)</p><br><p id='33' "
 "data-category='paragraph' style='font-size:16px'>\uf000 제3조(보험금의 지급사유)에서 "
 '장해지급률이 상해 발생<br>일부터 180일 이내에 확정되지 않는 경우에는 상해'),
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
