from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 증가된 위험과 관계없이 발생한 보험금</p><footer id='65' "
 "style='font-size:14px'>92</footer><p id='66' data-category='paragraph' "
 "style='font-size:20px'>지급사유에 관해서는 이를 원래대로 지급합니다.</p><br><p id='67' "
 "data-category='paragraph' style='font-size:16px'>\uf000 계약자 또는 피보험자가 고의 또는 "
 '중대한 과실로 제1항<br>각 호의 변경사실을 회사에 알리지 않았을 경우 변경후'),
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
