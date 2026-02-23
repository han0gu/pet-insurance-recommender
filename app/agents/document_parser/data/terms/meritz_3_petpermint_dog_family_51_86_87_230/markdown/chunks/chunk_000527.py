from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제4항 제2호 ㉠, ㉡ 또는 ㉤의 비용 : 비용의 전액을\n'
 '- 보상합니다.\n'
 '- ③ 제4항 제2호 ㉢ 또는 ㉣의 비용 : 이 비용과 제1호에\n'
 '- 따른 보상액의 합계액을 보험가입금액 한도내에서 보\n'
 '- 상합니다.\n'
 '# 제2조(보상하지 않는 손해)\uf000 회사는 아래의 사유를 원인으로 하여 생긴 배상책임을\n'
 '부담함으로써 입은 손해는 보상하지 않습니다.- ① 계약자 또는 피보험자(법인인 경우에는 그 이사 또는\n'
 '- 법인의 업무를 집행하는 그 밖의 기관)또는 이들의 법\n'
 '- 정대리인의 고의'),
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
