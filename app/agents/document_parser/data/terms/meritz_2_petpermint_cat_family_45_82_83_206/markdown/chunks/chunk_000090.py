from langchain_core.documents import Document

chunk = Document(
    page_content=('약자에게 보험증권 및 약관을 교부하고 변경된 계약자가 요\n'
 '청하는 경우 약관의 중요한 내용을 설명하여 드립니다.# 제24조(보험나이 등)\uf000 이 약관에서의 피보험자의 나이는 보험나이를 '
 '기준으로\n'
 '합니다. 다만, 제22조(계약의 무효) 제1항 제2호의 경우에\n'
 '는 실제 만 나이를 적용합니다.\n'
 '\uf000 제1항의 보험나이는 계약일 현재 피보험자의 실제 만 나\n'
 '이를 기준으로 6개월 미만의 끝수는 버리고 6개월 이상의\n'
 '끝수는 1년으로 하여 계산하며, 이후 매년 계약해당일에 나\n'
 '이가 증가하는 것으로 합니다.'),
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
