from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약자 또는 보험수익자가 회사에 알린 최종의 주소 또는 연락처로 등기우편 등 우편물\n'
 '- 에 대한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로 도달에 필요한 기간이\n'
 '- 지난 때에 계약자 또는 보험수익자에게 도달된 것으로 봅니다.\n'
 '# 제13조 (보험수익자의 지정)- ① 사망보험금의 경우는 보험수익자를 지정하지 않은 때에는 보험수익자를 피보험자의\n'
 '- 법정상속인, 기타 보험금의 경우는 피보험자로 합니다.\n'
 '- ② 제1항에 따라 지정된 보험수익자가 보험기간 중에 사망한 때에는 계약자는 다시 보험'),
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
