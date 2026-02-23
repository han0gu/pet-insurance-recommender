from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제5조(특별약관의 소멸)피보험자가 사망하였을경우에는 이 특별약관의 계약도 소멸되며 회사는 "보험료 및해약환급금 산출방법서"에서 정하는 '
 '바에 따라 피보험자의 사망 당시 이 특별약관의\n'
 '계약자적립액 및 미경과보험료를 계약자에게 지급합니다.- 86 -제6조(준용규정)'),
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
