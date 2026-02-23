from langchain_core.documents import Document

chunk = Document(
    page_content=('최초계약을 체결한 날부터 3년이 지났을 때<br>④ 회사가 이 계약을 청약할 때 피보험자의 건강상태를<br>판단할 수 있는 '
 '기초자료(건강진단서 사본 등)에 따라<br>승낙한 경우에 건강진단서 사본 등에 명기되어 있는<br>사항으로 보험금 지급사유가 발생하였을 '
 '때(계약자 또<br>는 피보험자가 회사에 제출한 기초자료의 내용 중 중<br>요사항을 고의로 사실과 다르게 작성한 때에는 '
 '계약을<br>해지할 수 있습니다)<br>⑤ 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회<br>를 주지 않았거나 계약자 또는 '
 '피보험자가'),
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
