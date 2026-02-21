from langchain_core.documents import Document

chunk = Document(
    page_content=('- 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보된 전자\n'
 '- 적 수단을 활용한 피보험자 의사표시의 확인방법 포함)\n'
 '- 3. 손해배상금 및 그 밖의 비용을 지급하였음을 명하는 서류\n'
 '- 4. 회사가 요구하는 그 밖의 서류\n'
 '# 제7조(보험금의 지급절차)- ① 회사는 제6조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 교부하고, 그\n'
 '- 서류를 접수받은 후 지체없이 지급할 보험금을 결정하고 지급할 보험금이 결정되면 7'),
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
