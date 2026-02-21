from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 피보험자가 회사에 제출한 기초자료의 내용 중 중\n'
 '- 요사항을 고의로 사실과 다르게 작성한 때에는 계약을\n'
 '- 해지할 수 있습니다)\n'
 '- ⑤ 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회\n'
 '- 를 주지 않았거나 계약자 또는 피보험자가 사실대로\n'
 '- 알리는 것을 방해한 경우, 계약자 또는 피보험자에게\n'
 '93사실대로 알리지 않게 하였거나 부실한 사항을 알릴\n'
 '것을 권유했을 때. 다만, 보험설계사 등의 행위가 없\n'
 '었다 하더라도 계약자 또는 피보험자가 사실대로 알리\n'
 '지 않거나 부실한 사항을 알렸다고 인정되는 경우에는'),
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
