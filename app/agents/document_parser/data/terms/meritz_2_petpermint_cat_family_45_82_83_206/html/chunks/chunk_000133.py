from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>【약관의 중요한 내용 예시】</h1><br><p id='86' "
 "data-category='list' style='font-size:16px'>- 청약의 철회에 관한 사항<br>- 지급한도, 면책사항, "
 '감액지급 사항 등 보험금 지급제<br>한 조건<br>- 계약 전 알릴 의무(고지의무) 위반의 효과<br>- 계약의 취소 및 무효에 관한 '
 '사항<br>- 해약환급금에 관한 사항<br>- 분쟁조정절차에 관한 사항<br>- 만기시 자동갱신되는 보험계약의 경우 자동갱신의 '
 '조건<br>- 저축성 보험계약의'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
