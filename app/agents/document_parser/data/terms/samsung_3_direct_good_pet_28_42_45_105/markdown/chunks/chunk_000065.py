from langchain_core.documents import Document

chunk = Document(
    page_content=('약관을 드리며, 전화를 이용하여 체결하는 계약은 계약자의 동의를 얻어 다음의 방법\n'
 '으로 약관의 중요한 내용을 설명할 수 있습니다.1. 전화를 이용하여 청약내용, 보험료납입, 보험기간, 계약 전 알릴 의무, 약관의 중요\n'
 '한 내용 등 계약을 체결하는 데 필요한 사항을 질문 또는 설명하는 방법. 이 경우\n'
 '계약자의 답변과 확인내용을 음성 녹음함으로써 약관의 중요한 내용을 설명한 것'),
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
