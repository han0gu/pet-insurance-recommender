from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해판정기준</h1><br><p id='18' data-category='list' style='font-size:16px'>1) "
 '청력장해는 순음청력검사 결과에 따라 데시벨(dB :<br>decibel)로서 표시하고 3회 이상 청력검사를 실시한<br>후 적용한다'),
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
