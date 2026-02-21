from langchain_core.documents import Document

chunk = Document(
    page_content=("3년간 행사하<br>지 않으면 소멸시효가 완성됩니다.</p><br><h1 id='30' "
 "style='font-size:18px'>【소멸시효】</h1><br><p id='31' data-category='paragraph' "
 "style='font-size:16px'>소멸시효는 해당 청구권을 행사할 수 있는 때부터 진행합<br>니다"),
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
