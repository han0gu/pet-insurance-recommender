from langchain_core.documents import Document

chunk = Document(
    page_content=('약도 소멸되며 회사는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 피보험자 또는 보험증권에 기재된 반려동물의 사망 당시 이 '
 '특별약관의 계약자적립액및 미경과보험료를 계약자에게 지급합니다.제8조(특별약관의 자동갱신)\uf000 이 특별약관의 【갱신계약】은 "제도성 '
 '특별약관 - 보장특약# 자동갱신(추가납입형) 특별약관"에 의해 계약자의 선택에 따라 자동갱신으로 운영합니다.\uf000 제1항에 의해 '
 '자동갱신을 적용할 경우 보험증권에 그 내용을 기재하여 드립니다.제9조(준용규정)'),
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
