from langchain_core.documents import Document

chunk = Document(
    page_content=('- 법규에서 정하는 바에 따릅니다.\n'
 '- ② 소득세법 등 관련법규가 제·개정 또는 폐지되는 경우 변경된 법령을 따릅니다.\n'
 '- 47 -<부표1> 보험금을 지급할 때의 적립이율 계산(보통약관 제9조 제4항 관련)| 구 분 | 기 간 | 지 급 이 자 |\n'
 '| --- | --- | --- |\n'
 '| 보장관련 보험금 | 지급기일의 다음 날부터 30일 이내 기간 | 보험계약대출이율 |\n'
 '| 보장관련 보험금 | 지급기일의 31일이후부터 60일이내 기간 | 보험계약대출이율+ 가산이율(4.0%) |'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
