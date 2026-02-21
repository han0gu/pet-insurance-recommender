from langchain_core.documents import Document

chunk = Document(
    page_content=('- 거나 약물을 주입하는 것\n'
 '\uf000 제1항의「수술」은 자택 등에서의 치료가 곤란하여 동물\n'
 '병원에서 행한 것에 한합니다.# 제4조(특별약관의 소멸)이 특별약관에서 정한 보상하는 손해가 더 이상 발생할 수\n'
 '없는 경우에는 이 특별약관은 그 때부터 소멸되며, 이 경우\n'
 '회사는「보험료 및 해약환급금 산출방법서」에서 정한 이\n'
 '특별약관의 그 때까지 적립한 계약자적립액 및 미경과보험\n'
 '료를 지급합니다.# 제5조(준용규정)이 특별약관에서 정하지 않은 사항은「반려동물 비용손해\n'
 '관련 특별약관 일반조항」을 따르고,「반려동물 비용손해'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
