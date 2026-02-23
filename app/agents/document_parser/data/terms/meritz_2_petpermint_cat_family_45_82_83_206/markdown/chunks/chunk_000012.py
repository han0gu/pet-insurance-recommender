from langchain_core.documents import Document

chunk = Document(
    page_content=('로【별표2(장해분류표)】에서 정한 장해지급률이 80%이상에\n'
 '해당하는 장해상태가 되었을 때에는 보험수익자에게 최초 1\n'
 '회에 한하여 이 보장의 보험가입금액 전액을 일반상해80%이\n'
 '상후유장해보험금으로 지급합니다.제4조(보험금 지급에 관한 세부규정)\uf000 제3조(보험금의 지급사유)에서 장해지급률이 상해 발생\n'
 '일부터 180일 이내에 확정되지 않는 경우에는 상해 발생일\n'
 '부터 180일이 되는 날의 의사 진단에 기초하여 고정될 것으\n'
 '로 인정되는 상태를 장해지급률로 결정합니다. 다만,【별표\n'
 '2(장해분류표)】에 장해판정시기를 별도로 정한 경우에는'),
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
