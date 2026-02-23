from langchain_core.documents import Document

chunk = Document(
    page_content=('급금 산출방법서" 에 정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약\n'
 '자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없\n'
 '습니다.# 제13조 (특별약관의 자동갱신)이 특별약관은 제도성 특별약관 4-1. [갱신형] 특별약관의 자동갱신 특별약관에 따라 갱\n'
 '신됩니다.# 제14조 (준용규정)이 특별약관에 정하지 않은 사항은 3-1. 반려견 의료비(치과및구강질환포함)(수술당일제\n'
 '외, 검사비포함)(재가입형) 특별약관을 따르며, 3-1. 반려견 의료비(치과및구강질환포함)('),
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
