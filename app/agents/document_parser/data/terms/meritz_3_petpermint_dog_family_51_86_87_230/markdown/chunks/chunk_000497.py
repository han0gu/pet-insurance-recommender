from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항 제2호의 경우에는 제3자로부터 손해의 배상을\n'
 '- 받을 수 있었던 금액\n'
 '- ③ 제1항 제3호의 경우에는 소송비용(중재 또는 조정에\n'
 '- 관한 비용 포함) 및 변호사비용과 회사의 동의를 받지\n'
 '- 않은 행위로 증가된 손해\n'
 '제9조(손해배상청구에 대한 회사의 해결)\uf000 피보험자가 피해자에게 손해배상책임을 지는 사고가 생\n'
 '긴 때에는 피해자는 이 특별약관에 따라 회사가 피보험자에\n'
 '게 지급책임을 지는 금액 한도내에서 회사에 대하여 보험금\n'
 '의 지급을 직접 청구할 수 있습니다. 그러나 회사는 피보험'),
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
