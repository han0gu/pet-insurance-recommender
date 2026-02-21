from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하여는 1년)이 지났을 때\n'
 '- 3. 최초계약을 체결한 날(재가입형 계약의 경우 최초 계약해당일을 말합니다)부터 3년\n'
 '- 이 지났을 때\n'
 '- 4. 회사가 이 계약을 청약할 때 반려견의 건강상태를 판단할 수 있는 기초자료(건강진\n'
 '- 단서 사본 등)에 따라 승낙한 경우에 건강진단서 사본 등에 명기되어 있는 사항으\n'
 '- 로 보험금 지급사유가 발생하였을 때(계약자 또는 피보험자가 회사에 제출한 기초\n'
 '- 자료의 내용 중 중요사항을 고의로 사실과 다르게 작성한 때에는 이 특별약관을\n'
 '- 해지할 수 있습니다)'),
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
