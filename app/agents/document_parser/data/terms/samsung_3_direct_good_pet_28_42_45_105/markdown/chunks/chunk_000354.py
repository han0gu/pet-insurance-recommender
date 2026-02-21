from langchain_core.documents import Document

chunk = Document(
    page_content=('- 금이 없거나 최초 가입할 때 안내한 해약환급금보다 적어질 수 있습니다.\n'
 '④ 회사는 제1항 제4호에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약관을 드리고, 변경된 계약자가 요청하는 경우 '
 '약관의 중요한 내용을 설명하여 드립니# 다.# 제18조 (보험나이 등)- ① 이 특별약관에서의 반려견의 나이는 만나이를 기준으로 '
 '합니다.\n'
 '- ② 제1항의 만나이는 계약일 현재 반려견의 실제 만나이를 기준으로 하며, 이후 매년 계\n'
 '- 약해당일에 나이가 증가하는 것으로 합니다.'),
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
