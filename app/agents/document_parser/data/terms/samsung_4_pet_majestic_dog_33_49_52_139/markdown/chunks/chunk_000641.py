from langchain_core.documents import Document

chunk = Document(
    page_content=('의 손해를 법적으로 보상해 주기 위해 법원에 납부하는 공탁금을 대신하는 보험상품을 공탁보증보\n'
 '험이라 하며, 이 보험에 가입하기 위해 필요한 보험료를 공탁보증보험료라 말합니다.# 제4조 (보상하지 않는 손해)① 회사는 피보험자가 '
 '아래와 같은 사유로 생긴 배상책임을 부담함으로써 입은 손해는- 120 -# 보상하지 않습니다.- 1. 계약자 및 피보험자, 이들의 가족 '
 '또는 사용인의 고의 또는 중대한 과실\n'
 '- 2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 이들과 유사한 사태'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
