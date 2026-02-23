from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>69</footer><p id='26' data-category='paragraph' "
 "style='font-size:18px'>사실과 다른 경우에는 정정된 나이 또는 성별에 해당하는<br>보험금 및 보험료로 "
 "변경합니다.</p><br><h1 id='27' style='font-size:16px'>【 보험나이 계산 예시 】</h1><br><p "
 "id='28' data-category='paragraph' style='font-size:18px'>생년월일 : 1988년 10월"),
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
