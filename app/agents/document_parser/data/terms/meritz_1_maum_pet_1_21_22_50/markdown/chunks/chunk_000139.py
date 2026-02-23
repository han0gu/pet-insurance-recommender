from langchain_core.documents import Document

chunk = Document(
    page_content=('보상책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 손해를 보상합니다. 이 특\n'
 '별약관과 다른 계약이 모두 의무보험인 경우에도 같습니다.# 손해액 ×# 이 계약의 보상책임액다른 계약이 없는 것으로 하여 각각 계산한 '
 '보상책임액의 합계액# 【사례】※ 보상책임액의 합계액이 손해액을 초과하는 경우 :\n'
 '계약A: 보상책임액 1,000만원 / 계약B: 보상책임액 1,000만원 / 손해액 : 1,000만원\n'
 '→ 계약A보험회사 : 500만원 지급 = 1,000만원 × 1,000만원 / (1,000만원 + 1,000\n'
 '만원)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
