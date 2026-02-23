from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 제3항에서 정한 보장개시일(책임개시일) 이후에 보험증권에 기재된 반려견에게 상\n'
 '- 해 또는 질병(이하 「사고」라 합니다)이 발생하여 그 치료를 직접적인 목적으로 국내\n'
 '- 에서 수의사에게 수술을 받은 경우 연간 2회에 한하여 피보험자가 부담한 수술 당일\n'
 '- 반려견의 치료에 사용된 비용(각종 할인 및 감면, 사후환급금액 등을 제외한 실수납액\n'
 '- 을 의미합니다. 이하「의료비」라 합니다)을 제5항에 따라 보험가입금액을 한도로 보\n'
 '- 험수익자에게 반려견 의료비(치과및구강질환포함) 보험금(수술당일)(이하 「의료비보'),
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
