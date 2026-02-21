from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제3조(피보험자의 범위)\uf000 이 특별약관에서 피보험자라 함은 아래에 정한 보험증권\n'
 '에 기재된 피보험자 및 그 가족을 말합니다.- ① 보험증권에 기재된 피보험자(이하「피보험자 본인」이\n'
 '- 라 합니다)\n'
 '- ② 피보험자 본인의 가족관계등록상 또는 주민등록상에\n'
 '- 기재된 배우자(이하「배우자」라 합니다)\n'
 '- ③ 피보험자 본인 또는 배우자와 생계를 같이 하는 동거\n'
 '- 친족 및 별거 중인 미혼자녀\n'
 '- \uf000 위 제1항에서 피보험자 본인과 본인 이외의 피보험자와\n'
 '- 의 관계는 사고발생 당시의 관계를 말합니다.'),
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
