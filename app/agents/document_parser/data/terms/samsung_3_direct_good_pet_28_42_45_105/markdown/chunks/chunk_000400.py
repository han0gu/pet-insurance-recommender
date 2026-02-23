from langchain_core.documents import Document

chunk = Document(
    page_content=('5. 손톱의 절제(며느리발톱의 제거 포함), 잔존유치, 잠복고환,\n'
 '제 3조 (보험금을 지급하지 않는 사유) 배꼽허니아(배꼽부위탈장), 항문낭 제거 등 건강동물에 실시하는 외과수술 및 기타\n'
 '① 회사는 아래의 사유로 보험금 지급사유가 발생한 때에는 보험금을 지급하지 않습니다. 검사 또는 손톱깎기 등의 처치비용\n'
 '1. 계약자 및 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실 6. 미용으로 인한 비용'),
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
