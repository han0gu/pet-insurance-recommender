from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>제2관 보험금의 지급</p><p id='30' data-category='paragraph' "
 "style='font-size:18px'>제3조(보험금의 지급사유)</p><br><p id='31' "
 "data-category='paragraph' style='font-size:18px'>회사는 보험증권에 기재된 피보험자가 보험기간 중에 "
 '상해<br>로【별표2(장해분류표)】에서 정한 장해지급률이 80%이상에<br>해당하는 장해상태가 되었을 때에는 보험수익자에게 최초 '
 '1<br>회에 한하여 이 보장의'),
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
