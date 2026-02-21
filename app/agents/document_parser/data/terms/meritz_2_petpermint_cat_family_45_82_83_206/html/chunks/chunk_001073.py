from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 판정기준</h1><br><p id='17' data-category='list' style='font-size:16px'>1) "
 '“심장 기능을 잃었을 때”라 함은 심장 이식을 한 경<br>우를 말한다.<br>2) “흉복부장기 또는 비뇨생식기 기능을 잃었을 때” '
 "라<br>함은 아래의 경우 중 하나에 해당하는 때를 말한다.</p><br><p id='18' data-category='list' "
 "style='font-size:20px'>가) 폐, 신장, 또는 간장의 장기이식을 한 경우<br>나) 장기이식을 하지 않고서는 생명유지가"),
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
