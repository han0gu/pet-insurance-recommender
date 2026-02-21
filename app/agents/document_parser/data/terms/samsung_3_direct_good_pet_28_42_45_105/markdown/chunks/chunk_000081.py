from langchain_core.documents import Document

chunk = Document(
    page_content=('- - 1988년 10월 2일\n'
 '- 32년 6개월 11일 = 33세\n'
 '[계약해당일 계산]최초계약일과 동일한 월, 일을 말합니다.\n'
 '계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일\n'
 '단, 계약해당일 2월 29일이 없을 경우에는 2월 28일을 계약해당일로 합니다.제23조 (계약의 소멸)① 회사가 제3조(보험금의 '
 '지급사유)에서 정한 상해 후유장해(80%이상) 보험금을 지급한\n'
 '때에는 그 손해보상의 원인이 생긴 때부터 이 계약은 소멸되며 그 때부터 효력이 없'),
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
