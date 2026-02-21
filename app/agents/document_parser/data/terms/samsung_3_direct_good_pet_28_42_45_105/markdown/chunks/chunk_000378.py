from langchain_core.documents import Document

chunk = Document(
    page_content=('- 발생시킨 경우\n'
 '- 2. 계약자 또는 피보험자가 보험금 청구에 관한 서류에 고의로 사실과 다른 것을 기재\n'
 '- 하였거나 그 서류 또는 증거를 위조 또는 변조한 경우. 다만, 이미 보험금 지급사\n'
 '- 유가 발생한 경우에는 보험금 지급에 영향을 미치지 않습니다.\n'
 '<용어풀이>[이미 발생한 보험금 지급사유에 대한 보험금 지급]계약자 또는 피보험자가 보험금 청구에 관한 서류를 변조하여 보험금을 청구한 '
 '경우, 회사는 그 사\n'
 '실을 안 날부터 1개월 이내에 계약을 해지할 수 있습니다. 다만, 이 경우에도 회사는 실제 발생한'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
