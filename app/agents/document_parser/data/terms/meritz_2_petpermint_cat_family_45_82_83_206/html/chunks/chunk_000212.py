from langchain_core.documents import Document

chunk = Document(
    page_content=('체납시 국세청 및 지방자치단체에<br>의해 채무자의 해약환급금이 압류될 수 있으며, 체납처<br>분 절차에 따라 회사는 채권자에게 '
 "해약환급금을 지급하<br>게 됩니다.</p><p id='78' data-category='paragraph' "
 "style='font-size:20px'>제6관 계약의 해지 및 해약환급금 등</p><p id='79' "
 "data-category='paragraph' style='font-size:18px'>제32조(계약자의 임의해지 및 피보험자의 서면동의 "
 "철회)</p><br><p id='80'"),
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
