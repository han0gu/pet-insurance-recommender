from langchain_core.documents import Document

chunk = Document(
    page_content=('- 자서명으로 동의를 얻어 수신확인을 조건으로 전자문서를 송신하여야 합니다. 계약자\n'
 '- 의 전자문서 수신이 확인되기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다.\n'
 '- 회사는 전자문서가 수신되지 않은 것을 확인한 경우에는 서면(등기우편 등)으로 다시\n'
 '- 알려드립니다.\n'
 '- ⑤ 제1항 제2호에 의한 계약의 해지가 보험금 지급사유 발생 후에 이루어진 경우에는 제\n'
 '- 제12조(계약 후 알릴 의무) 제4항 또는 제5항에 따라 보험금을 지급합니다.\n'
 '- ⑥ 제1항에도 불구하고 알릴 의무를 위반한 사실이 보험금 지급사유 발생에 영향을 미쳤'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
