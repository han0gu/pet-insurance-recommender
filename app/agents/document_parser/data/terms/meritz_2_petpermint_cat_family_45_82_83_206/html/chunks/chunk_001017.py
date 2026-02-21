from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 관절기능장해가 신경손<br>상으로 인한 경우에는 운동범위 측정이 아닌 근<br>력 및 근전도 검사를 기준으로 '
 "평가한다.</p><br><p id='31' data-category='paragraph' style='font-size:20px'>7) "
 "“관절 하나의 기능을 완전히 잃었을 때”라 함은 아<br>래의 경우 중 하나에 해당하는 경우를 말한다.</p><br><p id='32' "
 "data-category='list' style='font-size:20px'>가) 완전 강직(관절굳음)<br>나) 근전도 검사상"),
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
