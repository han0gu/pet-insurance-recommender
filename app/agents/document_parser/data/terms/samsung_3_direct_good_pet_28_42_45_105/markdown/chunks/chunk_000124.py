from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 회사는 계약과 관련된 개인정보를 안전하게 관리하여야 합니다.\n'
 '# 제43조 (준거법)이 계약은 대한민국 법에 따라 규율되고 해석되며, 약관에서 정하지 않은 사항은 「금융소\n'
 '비자 보호에 관한 법률」, 상법, 민법 등 관계 법령을 따릅니다.# 제44조 (예금보험에 의한 지급보장)회사가 파산 등으로 인하여 보험금 '
 '등을 지급하지 못할 경우에는 예금자보호법에서 정하\n'
 '는 바에 따라 그 지급을 보장합니다.<용어풀이># [예금자보호제도]예금자보호제도란 예금보험공사에서 금융기관 등으로부터 미리 보험료를 받아 '
 '적립해 두었다가 금'),
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
