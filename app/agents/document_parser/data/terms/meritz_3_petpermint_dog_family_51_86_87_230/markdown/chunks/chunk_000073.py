from langchain_core.documents import Document

chunk = Document(
    page_content=('에는 해당 문서를 드린 것으로 봅니다.- ① 서면교부\n'
 '- ② 우편 또는 전자우편\n'
 '- ③ 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시\n'
 '# 【약관의 중요한 내용 예시】- - 청약의 철회에 관한 사항\n'
 '- - 지급한도, 면책사항, 감액지급 사항 등 보험금 지급제\n'
 '- 한 조건\n'
 '- - 계약 전 알릴 의무(고지의무) 위반의 효과\n'
 '- - 계약의 취소 및 무효에 관한 사항\n'
 '- - 해약환급금에 관한 사항\n'
 '- - 분쟁조정절차에 관한 사항\n'
 '- - 만기시 자동갱신되는 보험계약의 경우 자동갱신의 조건\n'
 '- - 저축성 보험계약의 공시이율'),
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
