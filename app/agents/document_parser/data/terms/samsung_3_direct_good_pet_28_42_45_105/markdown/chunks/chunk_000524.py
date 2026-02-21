from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하시고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다.\n'
 '- 4. 사고증명서(진단서, 진료비계산서, 사망진단서, 장해진단서, 입원치료확인서, 의사\n'
 '- 처방전(처방조제비) 등)\n'
 '- 5. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이\n'
 '- 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이\n'
 '- 확보된 전자적 수단을 활용한 피보험자 의사표시의 확인방법 포함)\n'
 '- 6. 수탁기관 위탁비용 영수증 및 동물관리위탁업자가 제공하는 계약서(위탁관리업소'),
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
