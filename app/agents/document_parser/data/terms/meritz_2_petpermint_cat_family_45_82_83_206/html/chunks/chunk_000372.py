from langchain_core.documents import Document

chunk = Document(
    page_content=("id='20' data-category='paragraph' style='font-size:20px'>\uf000 제8항에 따라 "
 '계약자에게 해지된다는 사실을 알려드린<br>최초시점부터 90일 이내에 계약자의 재가입 의사가 확인되<br>지 않는 경우 해당 시점부터 '
 "계약은 해지됩니다.</p><br><p id='21' data-category='paragraph' "
 "style='font-size:20px'>\uf000 제6항에 따라 보험계약이 연장된 경우 계약자는 회사에<br>\uf000<br>재가입 "
 '의사를 표시할 수 있습니다'),
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
