from langchain_core.documents import Document

chunk = Document(
    page_content=('제외, 보험금 삭감, 보험료 할증 등)을 붙여 승낙할 수 있\n'
 '습니다.# 【 보험가입금액 제한 】피보험자가 가입을 할 수 있는 최대 보험가입금액을 제\n'
 '한하는 방법을 말합니다.# 【 일부보장 제외(부담보) 】일반적인 경우보다 위험이 높은 피보험자가 가입하기 위\n'
 '한 방법의 하나로, 특정 질병 또는 특정 신체 부위를 보\n'
 '장에서 제외하는 방법을 말합니다.# 【 보험금 삭감 】일반적인 경우보다 위험이 높은 피보험자가 가입하기 위\n'
 '한 방법의 하나로, 보험 가입 후 기간이 경과함에 따라\n'
 '위험의 크기 및 정도가 점차 감소하는 위험에 대해 적용'),
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
