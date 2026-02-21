from langchain_core.documents import Document

chunk = Document(
    page_content=('- ) 중에 제3항에서 정한 보장개시일(책임개시일) 이후에 보험증권에 기재된 반려묘에게\n'
 '- 상해 또는 질병(이하 「사고」 라 합니다)이 발생하여 그 치료를 직접적인 목적으로 국\n'
 '- 내에서 수의사에게 수술을 받은 경우 연간 2회에 한하여 피보험자가 부담한 수술 당\n'
 '- 일 반려묘의 치료에 사용된 비용(각종 할인 및 감면, 사후환급금액 등을 제외한 실수\n'
 '- 납액을 의미합니다. 이하 「의료비」 라 합니다)을 제4항에 따라 보험가입금액을 한도\n'
 '- 로 보험수익자에게 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관에서'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
