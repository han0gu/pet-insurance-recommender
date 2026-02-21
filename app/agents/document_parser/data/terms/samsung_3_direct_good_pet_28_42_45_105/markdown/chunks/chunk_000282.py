from langchain_core.documents import Document

chunk = Document(
    page_content=('검사비포함)(재가입형) 특별약관# 제 1조 (목적)이 특별약관은 보험계약자(이하「계약자」라 합니다)와 보험회사(이하「회사」라 합니다)\n'
 '사이에 보험증권에 기재된 반려견의 상해 또는 질병으로 인한 위험을 보장하기 위하여\n'
 '체결됩니다.# 제 2조 (용어의 정의)이 특별약관에서 사용되는 용어의 정의는 이 특별약관의 다른 조항에서 달리 정의되지\n'
 '않는 한 다음과 같습니다.# ① 계약 관련 용어- 1. 계약자: 회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사람을 말합니다.'),
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
