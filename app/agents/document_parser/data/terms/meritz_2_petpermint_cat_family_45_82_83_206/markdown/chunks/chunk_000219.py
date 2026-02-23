from langchain_core.documents import Document

chunk = Document(
    page_content=('개월 이내에 계약을 해지할 수 있습니다.- ① 계약자, 피보험자 또는 보험수익자가 보험금을 지급받\n'
 '- 을 목적으로 고의로 보험금 지급사유를 발생시킨 경\n'
 '- 우\n'
 '- ② 계약자, 피보험자 또는 보험수익자가 보험금 청구에\n'
 '- 관한 서류에 고의로 사실과 다른 것을 기재하였거나\n'
 '- 그 서류 또는 증거를 위조 또는 변조한 경우. 다만,\n'
 '- 이미 보험금 지급사유가 발생한 경우에는 이에 대한\n'
 '- 보험금은 지급합니다.\n'
 '# 【 예시 】입원특약에 가입한 피보험자가 20일간 입원하였음에도 불\n'
 '구하고 입원확인서를 변조하여 입원일수 30일에 해당하는'),
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
