from langchain_core.documents import Document

chunk = Document(
    page_content=('- 8. 위 제7호 이외의 방사선을 쬐는 것 또는 방사능 오염\n'
 '- 9. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태\n'
 '- 10. 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또는 급수 등 기본적인 관리에 대한 태만\n'
 '# 제3조(보험금의 청구)① 피보험자가 반려동물 사망위로금 특별약관 보험금을 청구할 때에는 다음의 서류를 회사에 제출하\n'
 '여야 합니다.- 1. 보험금 청구서(회사 양식)\n'
 '- 2. 사망을 확인할 수 있는 서류(동물폐사확인서, 동물화장증명서 등)'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
