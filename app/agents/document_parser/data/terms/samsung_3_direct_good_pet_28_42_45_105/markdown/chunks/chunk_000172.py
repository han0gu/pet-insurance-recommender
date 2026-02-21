from langchain_core.documents import Document

chunk = Document(
    page_content=('- 4. 회사가 이 특별약관을 청약할 때 피보험자의 건강상태를 판단할 수 있는 기초자료\n'
 '- (건강진단서 사본 등)에 따라 승낙한 경우에 건강진단서 사본 등에 명기되어 있는\n'
 '- 사항으로 보험금 지급사유가 발생하였을 때(계약자 또는 피보험자가 회사에 제출\n'
 '- 한 기초자료의 내용 중 중요사항을 고의로 사실과 다르게 작성한 때에는 특별약관\n'
 '- 을 해지할 수 있습니다)\n'
 '- 5. 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회를 주지 않았거나 계약자 또\n'
 '- 는 피보험자가 사실대로 알리는 것을 방해한 경우, 계약자 또는 피보험자에게 사실'),
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
