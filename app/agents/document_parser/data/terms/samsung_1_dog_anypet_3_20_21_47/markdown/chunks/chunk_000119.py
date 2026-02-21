from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 제1항 제2호의 경우에는 제3자로부터 손해의 배상을 받을 수 있었던 금액\n'
 '- 3. 제1항 제3호의 경우에는 소송비용(중재 또는 조정에 관한 비용 포함) 및 변호사비용과 회사의\n'
 '- 동의를 받지 않은 행위에 의하여 증가된 손해\n'
 '# 제8조(손해배상청구에 대한 회사의 해결)- ① 피보험자가 피해자에게 손해배상책임을 지는 사고가 생긴 때에는 피해자는 이 약관에 의하여 '
 '회사\n'
 '- 가 피보험자에게 지급책임을 지는 금액한도 내에서 회사에 대하여 보험금의 지급을 직접 청구할'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
