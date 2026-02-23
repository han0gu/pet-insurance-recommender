from langchain_core.documents import Document

chunk = Document(
    page_content=('- 유로 사망하였을 경우 회사는 "보험료 및 해약환급금 산출방법서"에서 정하는 바 약\n'
 '지급사유)에서 정한 무지개다리위로금(강아지, 사망)을제KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 117- 117 -# 에 따라 '
 '반려동물 사망 당시 이 특별약관의 계약자적립액 및 미경과보험료를 계# 약자에게 지급합니다.다수인 경우 제1항 내지 제2항은 보험의 '
 '목적별로 각각 적용합니다.# \uf000 보험의 목적이- 제6조(특별약관의 자동갱신)\n'
 '- \uf000 이 특별약관의 【갱신계약】은 "제도성 특별약관 - 보장특약 자동갱신(추가납입'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
