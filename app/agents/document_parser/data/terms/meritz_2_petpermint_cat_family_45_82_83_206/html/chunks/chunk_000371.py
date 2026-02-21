from langchain_core.documents import Document

chunk = Document(
    page_content=('보험계약이 연장된 경우 보험계약의 연장<br>일은 회사가 계약자의 재가입의사를 확인한 날(계약자 등이<br>회사에 보험금을 청구함으로써 '
 '계약자에게 연락이 닿아 회<br>사가 계약자의 재가입의사를 확인한 날 등)까지로 합니다.<br>회사는 계약자 등이 회사에 보험금을 '
 '청구하는 등 계약자에<br>게 연락이 닿으면 제4항의 내용과 90일 이내 계약자의 재가<br>입의사가 확인되지 않는 경우 계약이 '
 "해지된다는 사실을 알<br>려드립니다.</p><br><p id='20' data-category='paragraph'"),
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
