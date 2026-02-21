from langchain_core.documents import Document

chunk = Document(
    page_content=('험이라 하며, 이 보험에 가입하기 위해 필요한 보험료를 공탁보증보험료라 말합니다.# 제4조 (보상하지 않는 손해)① 회사는 피보험자가 '
 '아래와 같은 사유로 생긴 배상책임을 부담함으로써 입은 손해는- 87 -87 / 181# 보상하지 않습니다.- 1. 계약자 및 피보험자, '
 '이들의 가족 또는 사용인의 고의 또는 중대한 과실\n'
 '- 2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 이들과 유사한 사태\n'
 '- 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변'),
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
